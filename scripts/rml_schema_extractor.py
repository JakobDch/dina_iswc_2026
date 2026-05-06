"""
RML/R2RML Schema Extractor via Minimal Materialization

Extracts virtual schema from R2RML mappings by:
1. Parsing SQL schema from existing dump file
2. Creating SQLite with minimal dummy data
3. Running Morph-KGC materialization
4. Extracting schema from RDF output

Usage:
    python rml_schema_extractor.py <mapping.ttl> <ontology.owl> <sql_dump.sql> <output_dir>
"""

import argparse
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD

import morph_kgc


@dataclass
class TableSchema:
    """Schema of a SQL table."""
    name: str
    columns: List[Tuple[str, str]] = field(default_factory=list)  # (name, type)


@dataclass
class DataProperty:
    predicate_prefix: str
    predicate_local: str
    datatype: str


@dataclass
class ObjectProperty:
    predicate_prefix: str
    predicate_local: str
    target_prefix: str
    target_local: str


@dataclass
class SchemaClass:
    uri: URIRef
    prefix: str
    local_name: str
    data_properties: List[DataProperty] = field(default_factory=list)
    object_properties: List[ObjectProperty] = field(default_factory=list)


class SQLDumpParser:
    """Parses SQL dump and extracts schema + multiple rows of data per table."""

    MAX_ROWS_PER_TABLE = 100  # Limit to avoid memory issues

    def __init__(self, sql_file: str):
        import gzip
        if sql_file.endswith('.gz'):
            with gzip.open(sql_file, 'rt', encoding='utf-8', errors='ignore') as f:
                self.sql_content = f.read()
        else:
            with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
                self.sql_content = f.read()
        self.tables: Dict[str, TableSchema] = {}
        self.all_rows: Dict[str, List[List]] = {}  # table_name -> list of rows
        self.row_data: Dict[str, Dict[int, Dict[str, any]]] = {}  # table -> {row_id: {col: value}}

    def parse(self) -> Tuple[Dict[str, TableSchema], Dict[str, List[List]]]:
        """Parse CREATE TABLE, INSERT, UPDATE and COPY statements."""
        self._parse_create_tables()
        self._parse_inserts()
        self._parse_copy_statements()  # PostgreSQL COPY ... FROM stdin
        self._parse_updates()  # Handle INSERT + UPDATE pattern (like LUBM)
        self._convert_row_data_to_list()
        total_rows = sum(len(rows) for rows in self.all_rows.values())
        print(f"Parsed {len(self.tables)} tables, {total_rows} total rows")
        return self.tables, self.all_rows

    def _parse_create_tables(self):
        """Extract all table schemas from SQL dump."""
        pattern = r'CREATE\s+TABLE\s+[`"]?(\w+)[`"]?\s*\((.*?)\)(?:\s*ENGINE|\s*;|\s*$)'
        matches = re.findall(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)

        for table_name, columns_def in matches:
            table = TableSchema(name=table_name)

            for line in self._split_respecting_parens(columns_def):
                line = line.strip()
                if not line:
                    continue

                # Check if line STARTS with a constraint keyword (not a column definition)
                upper = line.upper().lstrip('`"')
                constraint_keywords = ['PRIMARY KEY', 'FOREIGN KEY', 'INDEX ', 'UNIQUE ', 'CONSTRAINT ', 'KEY `', 'KEY "']
                if any(upper.startswith(kw) for kw in constraint_keywords):
                    continue
                # Also skip standalone KEY definitions (MySQL index)
                if upper.startswith('KEY '):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    col_name = parts[0].strip('`"')
                    col_type = parts[1].upper().split('(')[0]
                    table.columns.append((col_name, col_type))

            if table.columns:
                self.tables[table_name.lower()] = table

    def _parse_inserts(self):
        """Extract INSERT rows for each table (up to MAX_ROWS_PER_TABLE)."""
        # Pattern for INSERT INTO table (cols) VALUES (...) - capture column list
        pattern = r'INSERT\s+(?:IGNORE\s+)?INTO\s+[`"]?(\w+)[`"]?\s*(?:\(([^)]+)\)\s*)?VALUES\s*'

        for match in re.finditer(pattern, self.sql_content, re.IGNORECASE):
            table_name = match.group(1).lower()

            if table_name not in self.all_rows:
                self.all_rows[table_name] = []

            # Skip if we already have enough rows
            if len(self.all_rows[table_name]) >= self.MAX_ROWS_PER_TABLE:
                continue

            # Parse column names if specified
            col_names = None
            if match.group(2):
                col_names = [c.strip().strip('`"').lower() for c in match.group(2).split(',')]

            # Find the VALUES part and extract ALL tuples
            start = match.end()
            all_values = self._extract_all_value_tuples(self.sql_content[start:])

            for values in all_values:
                if len(self.all_rows[table_name]) >= self.MAX_ROWS_PER_TABLE:
                    break

                if values and table_name in self.tables:
                    if col_names:
                        # Map values to correct column positions
                        table = self.tables[table_name]
                        full_row = [None] * len(table.columns)
                        col_index = {c[0].lower(): i for i, c in enumerate(table.columns)}
                        for i, col in enumerate(col_names):
                            if col in col_index and i < len(values):
                                full_row[col_index[col]] = values[i]
                        self.all_rows[table_name].append(full_row)
                    else:
                        self.all_rows[table_name].append(values)

    def _parse_copy_statements(self):
        """Parse PostgreSQL COPY ... FROM stdin statements."""
        # Pattern: COPY tablename (col1, col2, ...) FROM stdin;
        pattern = r'COPY\s+(\w+)\s*\(([^)]+)\)\s+FROM\s+stdin\s*;'

        for match in re.finditer(pattern, self.sql_content, re.IGNORECASE):
            table_name = match.group(1).lower()

            if table_name not in self.all_rows:
                self.all_rows[table_name] = []

            if len(self.all_rows[table_name]) >= self.MAX_ROWS_PER_TABLE:
                continue

            # Parse column names
            col_names = [c.strip().strip('"').lower() for c in match.group(2).split(',')]

            # Find the data section after the COPY statement
            start = match.end()
            # Data lines end with \. on a line by itself
            end_marker = self.sql_content.find('\n\\.', start)
            if end_marker == -1:
                continue

            data_section = self.sql_content[start:end_marker].strip()
            lines = data_section.split('\n')

            # Get all non-empty data lines (up to limit)
            for line in lines:
                if len(self.all_rows[table_name]) >= self.MAX_ROWS_PER_TABLE:
                    break

                line = line.strip()
                if not line:
                    continue

                # Tab-separated values in PostgreSQL COPY
                values = line.split('\t')

                if table_name in self.tables:
                    table = self.tables[table_name]
                    full_row = [None] * len(table.columns)
                    col_index = {c[0].lower(): i for i, c in enumerate(table.columns)}

                    for i, col in enumerate(col_names):
                        if col in col_index and i < len(values):
                            val = values[i]
                            # Handle PostgreSQL NULL (\N)
                            if val == '\\N':
                                full_row[col_index[col]] = None
                            else:
                                full_row[col_index[col]] = self._parse_value(val)

                    self.all_rows[table_name].append(full_row)

    def _extract_all_value_tuples(self, text: str) -> List[List]:
        """Extract all (value, value, ...) tuples from VALUES clause."""
        all_tuples = []
        if not text.startswith('('):
            return all_tuples

        values = []
        current = []
        depth = 0
        in_string = False
        string_char = None
        i = 1  # Skip opening paren

        while i < len(text) and len(all_tuples) < self.MAX_ROWS_PER_TABLE:
            char = text[i]

            # Handle string literals
            if char in ("'", '"') and (i == 0 or text[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                current.append(char)
            elif in_string:
                current.append(char)
            elif char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                if depth == 0:
                    # End of tuple
                    if current:
                        values.append(self._parse_value(''.join(current).strip()))
                    all_tuples.append(values)
                    values = []
                    current = []
                    # Look for next tuple or end
                    j = i + 1
                    while j < len(text) and text[j] in ' \t\n\r':
                        j += 1
                    if j < len(text) and text[j] == ',':
                        # More tuples follow
                        j += 1
                        while j < len(text) and text[j] in ' \t\n\r':
                            j += 1
                        if j < len(text) and text[j] == '(':
                            i = j  # Continue with next tuple
                        else:
                            break  # End of VALUES
                    else:
                        break  # End of VALUES (no more commas)
                else:
                    depth -= 1
                    current.append(char)
            elif char == ',' and depth == 0:
                values.append(self._parse_value(''.join(current).strip()))
                current = []
            else:
                current.append(char)
            i += 1

        return all_tuples

    def _parse_value(self, val: str) -> any:
        """Parse a SQL value to Python type."""
        if val.upper() == 'NULL':
            return None
        if val.startswith("'") and val.endswith("'"):
            return val[1:-1].replace("\\'", "'").replace("''", "'")
        if val.startswith('"') and val.endswith('"'):
            return val[1:-1]
        try:
            if '.' in val:
                return float(val)
            return int(val)
        except ValueError:
            return val

    def _split_respecting_parens(self, text: str) -> List[str]:
        """Split by comma respecting parentheses."""
        result = []
        current = []
        depth = 0
        for char in text:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                result.append(''.join(current))
                current = []
            else:
                current.append(char)
        if current:
            result.append(''.join(current))
        return result

    def _parse_updates(self):
        """Parse UPDATE statements to handle INSERT+UPDATE pattern (like LUBM)."""
        # Pattern: UPDATE table SET col=val WHERE col=val
        pattern = r'UPDATE\s+[`"]?(\w+)[`"]?\s+SET\s+([^;]+?)\s+WHERE\s+(\w+)\s*=\s*(\d+)'

        for match in re.finditer(pattern, self.sql_content, re.IGNORECASE):
            table_name = match.group(1).lower()
            if table_name not in self.tables:
                continue

            # Get WHERE clause info
            where_col = match.group(3).lower()
            where_val = int(match.group(4))

            table = self.tables[table_name]
            col_idx = None
            for i, (col_name, _) in enumerate(table.columns):
                if col_name.lower() == where_col:
                    col_idx = i
                    break

            if col_idx is None:
                continue

            # Parse SET assignments
            assignments = match.group(2)
            updates = {}
            for assignment in self._split_assignments(assignments):
                if '=' in assignment:
                    col, val = assignment.split('=', 1)
                    col = col.strip().strip('`"').lower()
                    updates[col] = self._parse_value(val.strip())

            # Apply updates to matching rows in all_rows
            if table_name in self.all_rows:
                for row in self.all_rows[table_name]:
                    if col_idx < len(row) and row[col_idx] == where_val:
                        # Found matching row - apply updates
                        for col, val in updates.items():
                            for i, (col_name, _) in enumerate(table.columns):
                                if col_name.lower() == col and i < len(row):
                                    row[i] = val
                                    break

            # Also track in row_data for rows not yet in all_rows
            if table_name not in self.row_data:
                self.row_data[table_name] = {}
            if where_val not in self.row_data[table_name]:
                self.row_data[table_name][where_val] = {}
            for col, val in updates.items():
                self.row_data[table_name][where_val][col] = val

    def _split_assignments(self, text: str) -> List[str]:
        """Split SET assignments by comma, respecting strings."""
        result = []
        current = []
        in_string = False
        string_char = None

        for i, char in enumerate(text):
            if char in ("'", '"') and (i == 0 or text[i-1] != ''):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                current.append(char)
            elif char == ',' and not in_string:
                result.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            result.append(''.join(current).strip())
        return result

    def _convert_row_data_to_list(self):
        """Convert row_data dict to all_rows list format for rows not yet captured."""
        for table_name, rows_by_id in self.row_data.items():
            if table_name not in self.tables:
                continue

            table = self.tables[table_name]
            if table_name not in self.all_rows:
                self.all_rows[table_name] = []

            # Find existing row IDs
            existing_ids = set()
            id_col_idx = 0  # Assume first column is ID
            for row in self.all_rows[table_name]:
                if row and len(row) > 0 and row[0] is not None:
                    existing_ids.add(row[0])

            # Add new rows from row_data
            for row_id, row_dict in rows_by_id.items():
                if row_id in existing_ids:
                    continue  # Already have this row
                if len(self.all_rows[table_name]) >= self.MAX_ROWS_PER_TABLE:
                    break

                # Build row in column order
                row = [None] * len(table.columns)
                row[0] = row_id  # Set the ID column
                for col_name, col_type in table.columns:
                    col_idx = None
                    for i, (cn, _) in enumerate(table.columns):
                        if cn.lower() == col_name.lower():
                            col_idx = i
                            break
                    if col_idx is not None:
                        val = row_dict.get(col_name.lower())
                        if val is not None:
                            row[col_idx] = val

                self.all_rows[table_name].append(row)


class CSVFolderParser:
    """Parses CSV files from a folder and extracts schema + multiple rows per table."""

    MAX_ROWS_PER_TABLE = 100  # Limit to avoid memory issues

    def __init__(self, csv_folder: str):
        self.csv_folder = csv_folder
        self.tables: Dict[str, TableSchema] = {}
        self.all_rows: Dict[str, List[List]] = {}

    def parse(self) -> Tuple[Dict[str, TableSchema], Dict[str, List[List]]]:
        """Parse all CSV files in the folder."""
        import csv as csv_module

        csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]

        for csv_file in csv_files:
            table_name = os.path.splitext(csv_file)[0].lower()
            filepath = os.path.join(self.csv_folder, csv_file)

            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv_module.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue

                # Create table schema - all columns are TEXT for CSV
                table = TableSchema(name=table_name.upper())
                for col in header:
                    table.columns.append((col.strip(), 'TEXT'))

                self.tables[table_name] = table

                # Get multiple data rows (up to MAX_ROWS_PER_TABLE)
                self.all_rows[table_name] = []
                for row in reader:
                    if len(self.all_rows[table_name]) >= self.MAX_ROWS_PER_TABLE:
                        break
                    self.all_rows[table_name].append(row)

                if not self.all_rows[table_name]:
                    self.all_rows[table_name] = [[None] * len(header)]

        total_rows = sum(len(rows) for rows in self.all_rows.values())
        print(f"Parsed {len(self.tables)} CSV files, {total_rows} total rows")
        return self.tables, self.all_rows


class SQLiteDatabaseCreator:
    """Creates SQLite database with multiple rows of real data per table."""

    def __init__(self, tables: Dict[str, TableSchema], all_rows: Dict[str, List[List]]):
        self.tables = tables
        self.all_rows = all_rows

    def create_database(self, db_path: str):
        """Create SQLite database with real data (multiple rows per table)."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables_created = 0
        rows_inserted = 0

        for table_name, table in self.tables.items():
            # Create table
            columns_def = []
            for col_name, col_type in table.columns:
                sqlite_type = self._mysql_to_sqlite_type(col_type)
                columns_def.append(f'"{col_name}" {sqlite_type}')

            create_sql = f'CREATE TABLE IF NOT EXISTS "{table.name}" ({", ".join(columns_def)})'
            try:
                cursor.execute(create_sql)
                tables_created += 1
            except sqlite3.Error as e:
                print(f"Warning: Could not create table {table.name}: {e}")
                continue

            # Insert all rows if available
            if table_name in self.all_rows:
                col_names = [f'"{c[0]}"' for c in table.columns]
                placeholders = ', '.join(['?' for _ in table.columns])
                insert_sql = f'INSERT INTO "{table.name}" ({", ".join(col_names)}) VALUES ({placeholders})'

                for values in self.all_rows[table_name]:
                    # Adjust values count to match columns
                    if len(values) >= len(table.columns):
                        values = list(values[:len(table.columns)])
                    else:
                        values = list(values) + [None] * (len(table.columns) - len(values))

                    # Clean values - replace None with empty string to avoid Morph-KGC bug
                    clean_values = []
                    for i, v in enumerate(values):
                        if v is None or (isinstance(v, str) and v.upper() == 'NULL'):
                            # Use empty string for TEXT, 0 for numeric types
                            col_type = table.columns[i][1].upper() if i < len(table.columns) else 'TEXT'
                            if any(t in col_type for t in ['INT', 'DECIMAL', 'FLOAT', 'DOUBLE', 'REAL', 'NUMERIC']):
                                clean_values.append(0)
                            else:
                                clean_values.append('')
                        elif isinstance(v, str) and v == 'None':
                            clean_values.append('')
                        else:
                            clean_values.append(v)

                    try:
                        cursor.execute(insert_sql, clean_values)
                        rows_inserted += 1
                    except sqlite3.Error as e:
                        print(f"Warning: Could not insert into {table.name}: {e}")

        conn.commit()
        conn.close()
        print(f"Created SQLite: {tables_created} tables, {rows_inserted} rows inserted")

    def _mysql_to_sqlite_type(self, mysql_type: str) -> str:
        """Convert MySQL type to SQLite type."""
        mysql_type = mysql_type.upper()
        if 'INT' in mysql_type:
            return 'INTEGER'
        if any(t in mysql_type for t in ['VARCHAR', 'CHAR', 'TEXT', 'ENUM']):
            return 'TEXT'
        if any(t in mysql_type for t in ['DECIMAL', 'FLOAT', 'DOUBLE', 'REAL']):
            return 'REAL'
        if 'DATE' in mysql_type or 'TIME' in mysql_type:
            return 'TEXT'
        return 'TEXT'


class MorphKGCMaterializer:
    """Uses Morph-KGC to materialize RDF."""

    def __init__(self, mapping_file: str, db_path: str):
        self.mapping_file = os.path.abspath(mapping_file)
        self.db_path = os.path.abspath(db_path)

    def materialize(self) -> Graph:
        """Run Morph-KGC and return resulting graph."""
        # Create config
        config = f"""
[CONFIGURATION]
output_format = N-TRIPLES

[DataSource]
mappings = {self.mapping_file}
db_url = sqlite:///{self.db_path}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config)
            config_path = f.name

        try:
            print("Running Morph-KGC materialization...")
            result = morph_kgc.materialize(config_path)

            g = Graph()
            if isinstance(result, str):
                g.parse(data=result, format='nt')
            else:
                g = result

            print(f"Materialized {len(g)} triples")
            return g

        except Exception as e:
            print(f"Materialization failed: {e}")
            return Graph()

        finally:
            try:
                os.unlink(config_path)
            except:
                pass


class SchemaExtractor:
    """Extracts schema from materialized RDF."""

    def __init__(self, rdf_graph: Graph, ontology_file: str, mapping_file: str = None):
        self.rdf = rdf_graph
        self.ontology = self._load_ontology(ontology_file)
        self.classes: Dict[URIRef, SchemaClass] = {}
        self.prefixes: Dict[str, str] = {}

        # Load prefixes from mapping file (most authoritative)
        if mapping_file:
            mapping_graph = Graph()
            try:
                mapping_graph.parse(mapping_file, format='turtle')
                for prefix, ns in mapping_graph.namespaces():
                    if prefix:
                        self.prefixes[prefix] = str(ns)
            except:
                pass

        # Add prefixes from ontology
        for prefix, ns in self.ontology.namespaces():
            if prefix and prefix not in self.prefixes:
                self.prefixes[prefix] = str(ns)

        # Add prefixes from RDF graph
        for prefix, ns in self.rdf.namespaces():
            if prefix and prefix not in self.prefixes:
                self.prefixes[prefix] = str(ns)

        self.namespace_to_prefix = {v: k for k, v in self.prefixes.items()}

    def _load_ontology(self, ontology_file: str) -> Graph:
        g = Graph()
        if not ontology_file:
            return g
        try:
            g.parse(ontology_file, format='turtle')
        except:
            try:
                g.parse(ontology_file, format='xml')
            except:
                print(f"Warning: Could not parse ontology")
        return g

    def extract(self) -> Dict[URIRef, SchemaClass]:
        """Extract schema from RDF graph."""
        # Find all instances with rdf:type
        for subj, _, cls_uri in self.rdf.triples((None, RDF.type, None)):
            if not isinstance(cls_uri, URIRef):
                continue
            if str(cls_uri).startswith(str(RDF)) or str(cls_uri).startswith(str(OWL)):
                continue

            self._ensure_class(cls_uri)

            # Find all properties of this instance
            for _, pred, obj in self.rdf.triples((subj, None, None)):
                if pred == RDF.type:
                    continue

                pred_prefix, pred_local = self._get_prefix_local(pred)

                if isinstance(obj, URIRef):
                    # Object property
                    target_type = self.rdf.value(obj, RDF.type)
                    if target_type and isinstance(target_type, URIRef):
                        target_prefix, target_local = self._get_prefix_local(target_type)
                        self.classes[cls_uri].object_properties.append(
                            ObjectProperty(pred_prefix, pred_local, target_prefix, target_local)
                        )
                else:
                    # Data property
                    datatype = 'string'
                    if hasattr(obj, 'datatype') and obj.datatype:
                        dt = str(obj.datatype)
                        datatype = dt.split('#')[-1] if '#' in dt else dt.split('/')[-1]

                    self.classes[cls_uri].data_properties.append(
                        DataProperty(pred_prefix, pred_local, datatype)
                    )

        # Deduplicate
        for cls in self.classes.values():
            cls.data_properties = self._dedupe_data(cls.data_properties)
            cls.object_properties = self._dedupe_obj(cls.object_properties)

        # Derive superclass models from ontology hierarchy
        self._derive_superclass_models()

        return self.classes

    def _ensure_class(self, uri: URIRef):
        if uri not in self.classes:
            prefix, local = self._get_prefix_local(uri)
            self.classes[uri] = SchemaClass(uri, prefix, local)

    def _get_all_subclasses(self, superclass: URIRef) -> Set[URIRef]:
        """Get all transitive subclasses of a given class from the ontology."""
        subclasses = set()
        # Direct subclasses: ?sub rdfs:subClassOf ?superclass
        for sub, _, _ in self.ontology.triples((None, RDFS.subClassOf, superclass)):
            if isinstance(sub, URIRef):
                subclasses.add(sub)
                # Recurse for transitive subclasses
                subclasses.update(self._get_all_subclasses(sub))
        return subclasses

    def _derive_superclass_models(self):
        """Derive class_semantic_models for superclasses defined in the ontology.

        For each superclass (e.g. Learner, Educator, Faculty, Person, Course),
        collect the union of properties from all materialized subclasses and
        create a SchemaClass entry. Object property targets are also lifted
        to superclass level where applicable.
        """
        # Collect all superclasses referenced in ontology (via rdfs:subClassOf)
        all_superclasses: Set[URIRef] = set()
        for _, _, sup in self.ontology.triples((None, RDFS.subClassOf, None)):
            if isinstance(sup, URIRef):
                all_superclasses.add(sup)

        # Process from most specific to most general (bottom-up)
        # by sorting: superclasses with fewer transitive subclasses first
        ordered = sorted(all_superclasses, key=lambda s: len(self._get_all_subclasses(s)))

        for sup_uri in ordered:
            if sup_uri in self.classes:
                continue  # Already materialized directly

            subclasses = self._get_all_subclasses(sup_uri)
            # Only create if at least one subclass was materialized
            materialized_subs = [sc for sc in subclasses if sc in self.classes]
            if not materialized_subs:
                continue

            self._ensure_class(sup_uri)
            sup_cls = self.classes[sup_uri]

            # Union of all subclass properties
            for sub_uri in materialized_subs:
                sub_cls = self.classes[sub_uri]

                for dp in sub_cls.data_properties:
                    sup_cls.data_properties.append(dp)

                for op in sub_cls.object_properties:
                    # Lift object property target to superclass if applicable
                    # e.g. if BachelorCandidate.mentor -> SeniorScholar,
                    # Learner.mentor should -> Educator (if SeniorScholar subClassOf Educator)
                    lifted_op = self._lift_object_property_target(op)
                    sup_cls.object_properties.append(lifted_op)

            sup_cls.data_properties = self._dedupe_data(sup_cls.data_properties)
            sup_cls.object_properties = self._dedupe_obj(sup_cls.object_properties)

            print(f"Derived superclass {sup_cls.local_name}: "
                  f"{len(sup_cls.data_properties)} data, "
                  f"{len(sup_cls.object_properties)} object properties "
                  f"(from {len(materialized_subs)} subclasses)")

    def _lift_object_property_target(self, op: ObjectProperty) -> ObjectProperty:
        """Try to lift an object property target to its nearest superclass
        that has materialized subclasses.

        E.g. if target is SeniorScholar and Educator is a superclass with
        multiple materialized subclasses, lift to Educator.
        """
        # Reconstruct target URI from prefix:local
        target_ns = self.prefixes.get(op.target_prefix, '')
        if not target_ns:
            return op
        target_uri = URIRef(target_ns + op.target_local)

        # Walk up the hierarchy to find the most general superclass
        # that still has materialized subclasses
        best_super = None
        for _, _, sup in self.ontology.triples((target_uri, RDFS.subClassOf, None)):
            if not isinstance(sup, URIRef):
                continue
            # Check if this superclass has multiple materialized subclasses
            sub_of_sup = self._get_all_subclasses(sup)
            materialized = [s for s in sub_of_sup if s in self.classes]
            if len(materialized) >= 2:
                best_super = sup

        if best_super and best_super != target_uri:
            sup_prefix, sup_local = self._get_prefix_local(best_super)
            return ObjectProperty(op.predicate_prefix, op.predicate_local,
                                  sup_prefix, sup_local)
        return op

    def _get_prefix_local(self, uri: URIRef) -> Tuple[str, str]:
        uri_str = str(uri)
        for ns, prefix in self.namespace_to_prefix.items():
            if uri_str.startswith(ns):
                return prefix, uri_str[len(ns):]
        if '#' in uri_str:
            return '', uri_str.rsplit('#', 1)[1]
        return '', uri_str.rsplit('/', 1)[-1]

    def _dedupe_data(self, props):
        seen = set()
        result = []
        for p in props:
            key = (p.predicate_prefix, p.predicate_local)
            if key not in seen:
                seen.add(key)
                result.append(p)
        return sorted(result, key=lambda x: x.predicate_local)

    def _dedupe_obj(self, props):
        seen = set()
        result = []
        for p in props:
            key = (p.predicate_prefix, p.predicate_local, p.target_prefix, p.target_local)
            if key not in seen:
                seen.add(key)
                result.append(p)
        return sorted(result, key=lambda x: x.predicate_local)


class SchemaTTLGenerator:
    """Generates TTL files."""

    def __init__(self, classes: Dict[URIRef, SchemaClass], prefixes: Dict[str, str]):
        self.classes = classes
        self.prefixes = prefixes

    def generate_all(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        generated = 0

        for cls in sorted(self.classes.values(), key=lambda x: x.local_name):
            if not cls.data_properties and not cls.object_properties:
                continue

            filepath = os.path.join(output_dir, f"{cls.local_name}.ttl")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self._generate_ttl(cls))

            print(f"Generated {cls.local_name}.ttl: {len(cls.data_properties)} data, {len(cls.object_properties)} object")
            generated += 1

        print(f"\nGenerated {generated} schema files")

    def _generate_ttl(self, cls: SchemaClass) -> str:
        lines = ['@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .']

        used = {cls.prefix}
        for p in cls.data_properties:
            used.add(p.predicate_prefix)
        for p in cls.object_properties:
            used.add(p.predicate_prefix)
            used.add(p.target_prefix)

        for prefix in sorted(used):
            if prefix in self.prefixes and prefix != 'xsd':
                lines.append(f'@prefix {prefix}: <{self.prefixes[prefix]}> .')
        lines.append('')

        prop_lines = []
        for p in cls.data_properties:
            prop_lines.append(f'    {p.predicate_prefix}:{p.predicate_local} xsd:{p.datatype}')
        for p in cls.object_properties:
            prop_lines.append(f'    {p.predicate_prefix}:{p.predicate_local} {p.target_prefix}:{p.target_local}')

        if prop_lines:
            lines.append(f'{cls.prefix}:{cls.local_name} {prop_lines[0].strip()} ;')
            for pl in prop_lines[1:-1]:
                lines.append(pl + ' ;')
            if len(prop_lines) > 1:
                lines.append(prop_lines[-1] + ' .')
            else:
                lines[-1] = lines[-1].replace(' ;', ' .')

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Extract schema from R2RML via materialization')
    parser.add_argument('mapping_file', help='R2RML mapping (.ttl)')
    parser.add_argument('ontology_file', nargs='?', default=None, help='Ontology (.owl/.ttl) - optional')
    parser.add_argument('data_source', help='SQL dump file or CSV folder')
    parser.add_argument('output_dir', help='Output directory')

    args = parser.parse_args()

    if not os.path.isfile(args.mapping_file):
        print(f"Error: Mapping file not found: {args.mapping_file}")
        return 1

    is_csv_folder = os.path.isdir(args.data_source)
    if not is_csv_folder and not os.path.isfile(args.data_source):
        print(f"Error: Data source not found: {args.data_source}")
        return 1

    if args.ontology_file and not os.path.isfile(args.ontology_file):
        print(f"Warning: Ontology file not found: {args.ontology_file}, proceeding without")
        args.ontology_file = None

    print(f"Mapping: {args.mapping_file}")
    print(f"Ontology: {args.ontology_file or '(none)'}")
    print(f"Data Source: {args.data_source} ({'CSV folder' if is_csv_folder else 'SQL dump'})")
    print()

    # Step 1: Parse schema
    if is_csv_folder:
        print("Step 1: Parsing CSV files...")
        data_parser = CSVFolderParser(args.data_source)
    else:
        print("Step 1: Parsing SQL schema...")
        data_parser = SQLDumpParser(args.data_source)
    tables, all_rows = data_parser.parse()

    # Step 2: Create SQLite database
    print("\nStep 2: Creating SQLite database...")
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        db_creator = SQLiteDatabaseCreator(tables, all_rows)
        db_creator.create_database(db_path)

        # Step 3: Materialize
        print("\nStep 3: Materializing RDF...")
        materializer = MorphKGCMaterializer(args.mapping_file, db_path)
        rdf_graph = materializer.materialize()

        if len(rdf_graph) == 0:
            print("Error: No triples materialized")
            return 1

        # Step 4: Extract schema
        print("\nStep 4: Extracting schema...")
        extractor = SchemaExtractor(rdf_graph, args.ontology_file, args.mapping_file)
        classes = extractor.extract()
        print(f"Found {len(classes)} classes")

        # Step 5: Generate TTL
        print("\nStep 5: Generating schema files...")
        generator = SchemaTTLGenerator(classes, extractor.prefixes)
        generator.generate_all(args.output_dir)

    finally:
        try:
            os.unlink(db_path)
        except:
            pass

    return 0


if __name__ == '__main__':
    exit(main())