"""Tests for retrieval ground truth extraction and metrics."""

import pytest

from src.evaluation.retrieval_ground_truth import (
    RetrievalGroundTruth,
    RetrievalNecessity,
    InstanceReference,
    RetrievalSchemaGT,
    extract_retrieval_ground_truth,
    extract_retrieval_gt_for_query,
)
from src.evaluation.retrieval_metrics import (
    RetrievalMetricsResult,
    parse_retrieved_triples,
    calculate_retrieval_metrics,
)


# ============================================================================
# GT Extraction Tests
# ============================================================================


class TestBasicExtraction:
    """Test extraction of classes and properties from GT SPARQL."""

    def test_basic_bgp_extraction(self):
        """Required BGP triples yield REQUIRED classes and properties."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?student ?course WHERE {
            ?student a eduo:Student .
            ?student eduo:takesCourse ?course .
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.is_valid
        assert "http://example.org/ontology/education#Student" in gt.required_classes
        assert "http://example.org/ontology/education#takesCourse" in gt.required_properties

    def test_rdf_type_excluded_from_properties(self):
        """rdf:type is never added as a property."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?x WHERE { ?x a eduo:Student . }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.is_valid
        assert "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" not in gt.all_properties

    def test_multiple_classes_and_properties(self):
        """Extract multiple classes and properties from a complex query."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?student ?course ?dept WHERE {
            ?student a eduo:Student .
            ?student eduo:takesCourse ?course .
            ?student eduo:memberOf ?dept .
            ?dept a eduo:Department .
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.is_valid
        assert "http://example.org/ontology/education#Student" in gt.required_classes
        assert "http://example.org/ontology/education#Department" in gt.required_classes
        assert "http://example.org/ontology/education#takesCourse" in gt.required_properties
        assert "http://example.org/ontology/education#memberOf" in gt.required_properties

    def test_parse_error_handled(self):
        """Malformed SPARQL sets parse_error."""
        gt = extract_retrieval_ground_truth("NOT SPARQL AT ALL")
        assert not gt.is_valid
        assert gt.parse_error is not None


class TestOptionalHandling:
    """Test OPTIONAL -> ACCEPTABLE classification."""

    def test_optional_yields_acceptable(self):
        """Triples in OPTIONAL blocks are ACCEPTABLE."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?student ?name WHERE {
            ?student a eduo:Student .
            OPTIONAL { ?student eduo:name ?name }
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.classes["http://example.org/ontology/education#Student"] == RetrievalNecessity.REQUIRED
        assert gt.properties["http://example.org/ontology/education#name"] == RetrievalNecessity.ACCEPTABLE

    def test_nested_optional(self):
        """Nested OPTIONAL inherits acceptable status."""
        sparql = """
        PREFIX eno: <http://example.org/ontology/energy#>
        SELECT ?field ?operator ?name WHERE {
            ?field a eno:Field .
            OPTIONAL {
                ?field eno:currentFieldOperator ?operator .
                OPTIONAL { ?operator eno:name ?name }
            }
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.properties["http://example.org/ontology/energy#currentFieldOperator"] == RetrievalNecessity.ACCEPTABLE
        assert gt.properties["http://example.org/ontology/energy#name"] == RetrievalNecessity.ACCEPTABLE

    def test_required_wins_over_acceptable(self):
        """If property appears in both required and optional, it's REQUIRED."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?x ?y ?z ?w WHERE {
            ?x eduo:name ?y .
            OPTIONAL { ?z eduo:name ?w }
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.properties["http://example.org/ontology/education#name"] == RetrievalNecessity.REQUIRED


class TestMinusHandling:
    """Test MINUS -> REQUIRED classification."""

    def test_minus_yields_required(self):
        """Triples in MINUS blocks are REQUIRED (central to query semantics)."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?prof WHERE {
            ?prof a eduo:Professor .
            ?prof eduo:teacherOf ?course .
            MINUS { ?student eduo:advisor ?prof }
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.properties["http://example.org/ontology/education#teacherOf"] == RetrievalNecessity.REQUIRED
        assert gt.properties["http://example.org/ontology/education#advisor"] == RetrievalNecessity.REQUIRED


class TestUnionHandling:
    """Test UNION -> both sides REQUIRED."""

    def test_union_both_required(self):
        """Both sides of UNION are REQUIRED."""
        sparql = """
        PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
        SELECT ?product ?price WHERE {
            { ?offer bsbm:price ?price }
            UNION
            { ?offer bsbm:product ?product }
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.properties["http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/price"] == RetrievalNecessity.REQUIRED
        assert gt.properties["http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/product"] == RetrievalNecessity.REQUIRED


class TestFilterInExtraction:
    """Test FILTER(?var IN (...)) class extraction."""

    def test_filter_in_extracts_classes(self):
        """FILTER(?var IN (Class1, Class2)) with rdf:type ?var extracts classes."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?course WHERE {
            ?course a ?courseType .
            FILTER(?courseType IN (eduo:GraduateCourse, eduo:UndergraduateCourse))
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.is_valid
        assert "http://example.org/ontology/education#GraduateCourse" in gt.required_classes
        assert "http://example.org/ontology/education#UndergraduateCourse" in gt.required_classes


class TestSubqueryExtraction:
    """Test elements inside subqueries are extracted."""

    def test_subquery_extraction(self):
        """Elements inside subqueries (ToMultiSet) are extracted."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?dept ?count WHERE {
            { SELECT ?dept (COUNT(?student) AS ?count) WHERE {
                ?dept a eduo:Department .
                ?student eduo:memberOf ?dept .
                ?student a eduo:Student .
            } GROUP BY ?dept }
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.is_valid
        assert "http://example.org/ontology/education#Department" in gt.required_classes
        assert "http://example.org/ontology/education#Student" in gt.required_classes
        assert "http://example.org/ontology/education#memberOf" in gt.required_properties


class TestInstanceExtraction:
    """Test extraction of data instances from SPARQL queries."""

    def test_uri_constant_instance(self):
        """URI constants in BGP are extracted as instances."""
        sparql = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?dept WHERE {
            ?dept eduo:subOrganizationOf <http://www.university0.edu> .
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.is_valid
        instance_values = {inst.value for inst in gt.instances}
        assert "http://www.university0.edu" in instance_values
        inst = next(i for i in gt.instances if i.value == "http://www.university0.edu")
        assert inst.value_type == "uri"
        assert inst.associated_property == "http://example.org/ontology/education#subOrganizationOf"

    def test_string_filter_instance(self):
        """String literals in FILTER are extracted as instances."""
        sparql = """
        PREFIX eno: <http://example.org/ontology/energy#>
        SELECT ?field WHERE {
            ?field a eno:Field .
            ?field eno:name ?name .
            FILTER(UCASE(?name) = "TROLL")
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        assert gt.is_valid
        instance_values = {inst.value for inst in gt.instances}
        assert "TROLL" in instance_values
        inst = next(i for i in gt.instances if i.value == "TROLL")
        assert inst.value_type == "string"

    def test_regex_filter_instance(self):
        """REGEX pattern strings are extracted as instances."""
        sparql = """
        PREFIX eno: <http://example.org/ontology/energy#>
        SELECT ?field WHERE {
            ?field a eno:Field .
            ?company eno:name ?companyName .
            FILTER(REGEX(?companyName, "STATOIL", "i"))
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        instance_values = {inst.value for inst in gt.instances}
        assert "STATOIL" in instance_values

    def test_numeric_filter_instance(self):
        """Numeric values in FILTER are extracted as instances."""
        sparql = """
        PREFIX eno: <http://example.org/ontology/energy#>
        SELECT ?wellbore WHERE {
            ?wellbore a eno:Wellbore .
            ?wellbore eno:wellboreTotalDepth ?depth .
            FILTER(?depth > 3000)
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        instance_values = {inst.value for inst in gt.instances}
        assert "3000" in instance_values
        inst = next(i for i in gt.instances if i.value == "3000")
        assert inst.value_type == "numeric"

    def test_time_filter_instance(self):
        """Time values in FILTER are extracted as instances."""
        sparql = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip WHERE {
            ?stopTime a trn:StopTime .
            ?stopTime trn:trip ?trip .
            ?stopTime trn:departureTime ?departureTime .
            FILTER(?departureTime >= "00:00:00" && ?departureTime < "01:00:00")
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        instance_values = {inst.value for inst in gt.instances}
        assert "00:00:00" in instance_values
        assert "01:00:00" in instance_values
        inst = next(i for i in gt.instances if i.value == "00:00:00")
        assert inst.value_type == "temporal"

    def test_instance_variable_resolution(self):
        """Instance's associated class/property are resolved via variable tracing."""
        sparql = """
        PREFIX eno: <http://example.org/ontology/energy#>
        SELECT ?field WHERE {
            ?field a eno:Field .
            ?field eno:name ?name .
            FILTER(UCASE(?name) = "TROLL")
        }
        """
        gt = extract_retrieval_ground_truth(sparql)
        inst = next(i for i in gt.instances if i.value == "TROLL")
        # ?name is connected via ?field eno:name ?name, where ?field is a eno:Field
        assert inst.associated_property == "http://example.org/ontology/energy#name"
        assert inst.associated_class == "http://example.org/ontology/energy#Field"


class TestCorpusIntegration:
    """Integration tests using actual experimental corpus queries."""

    def test_base01_full_extraction(self):
        """BASE01: Students taking courses from own department."""
        from data.queries.experimental_corpus import BASE01

        gt = extract_retrieval_gt_for_query(BASE01)
        assert gt.is_valid
        # Required classes
        assert "http://example.org/ontology/education#Student" in gt.required_classes
        # Required properties
        assert "http://example.org/ontology/education#takesCourse" in gt.required_properties
        assert "http://example.org/ontology/education#memberOf" in gt.required_properties
        assert "http://example.org/ontology/education#teacherOf" in gt.required_properties
        assert "http://example.org/ontology/education#worksFor" in gt.required_properties
        # Acceptable properties (from OPTIONAL blocks)
        assert gt.properties["http://example.org/ontology/education#name"] == RetrievalNecessity.ACCEPTABLE

    def test_base09_minus_required(self):
        """BASE09: Professors without advisees - MINUS property is REQUIRED."""
        from data.queries.experimental_corpus import BASE09

        gt = extract_retrieval_gt_for_query(BASE09)
        assert gt.is_valid
        assert gt.properties["http://example.org/ontology/education#advisor"] == RetrievalNecessity.REQUIRED
        assert gt.properties["http://example.org/ontology/education#teacherOf"] == RetrievalNecessity.REQUIRED
        assert "http://example.org/ontology/education#Professor" in gt.required_classes
        assert "http://example.org/ontology/education#GraduateCourse" in gt.required_classes

    def test_base12_filter_in(self):
        """BASE12: FILTER IN pattern extracts both course types."""
        from data.queries.experimental_corpus import BASE12

        gt = extract_retrieval_gt_for_query(BASE12)
        assert gt.is_valid
        assert "http://example.org/ontology/education#GraduateCourse" in gt.all_classes
        assert "http://example.org/ontology/education#UndergraduateCourse" in gt.all_classes

    def test_base13_troll_instance(self):
        """BASE13: TROLL field - instance extracted."""
        from data.queries.experimental_corpus import BASE13

        gt = extract_retrieval_gt_for_query(BASE13)
        assert gt.is_valid
        instance_values = {inst.value for inst in gt.instances}
        assert "TROLL" in instance_values

    def test_base14_name_instance(self):
        """BASE14: University0 name instance extracted (via eduo:name, not URI)."""
        from data.queries.experimental_corpus import BASE14

        gt = extract_retrieval_gt_for_query(BASE14)
        assert gt.is_valid
        instance_values = {inst.value for inst in gt.instances}
        assert "University0" in instance_values

    def test_base16_statoil_instance(self):
        """BASE16: STATOIL regex instance extracted."""
        from data.queries.experimental_corpus import BASE16

        gt = extract_retrieval_gt_for_query(BASE16)
        assert gt.is_valid
        instance_values = {inst.value for inst in gt.instances}
        assert "STATOIL" in instance_values

    def test_base18_numeric_instance(self):
        """BASE18: Depth > 3000 numeric instance extracted."""
        from data.queries.experimental_corpus import BASE18

        gt = extract_retrieval_gt_for_query(BASE18)
        assert gt.is_valid
        instance_values = {inst.value for inst in gt.instances}
        assert "3000" in instance_values

    def test_all_parseable_queries_have_gt(self):
        """All parseable queries produce non-empty GT."""
        from data.queries.experimental_corpus import ALL_EXPERIMENTAL_QUERIES

        for q in ALL_EXPERIMENTAL_QUERIES:
            gt = extract_retrieval_gt_for_query(q)
            if gt.is_valid:
                assert len(gt.all_classes) > 0 or len(gt.all_properties) > 0, (
                    f"{q.query_id} parsed but has no classes or properties"
                )


# ============================================================================
# Turtle Parsing Tests
# ============================================================================


class TestParseRetrievedTriples:
    """Test parsing of agent's Turtle-format retrieved triples."""

    def test_basic_turtle_parsing(self):
        """Parse simple Turtle and extract classes and properties."""
        turtle = """
        @prefix eduo: <http://example.org/ontology/education#> .
        eduo:Student eduo:takesCourse eduo:Course .
        """
        classes, properties = parse_retrieved_triples([turtle])
        assert "http://example.org/ontology/education#Student" in classes
        assert "http://example.org/ontology/education#Course" in classes
        assert "http://example.org/ontology/education#takesCourse" in properties

    def test_subclass_excluded_from_properties(self):
        """rdfs:subClassOf is NOT included as a property."""
        turtle = """
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix eduo: <http://example.org/ontology/education#> .
        eduo:DoctoralCandidate rdfs:subClassOf eduo:Student .
        """
        classes, properties = parse_retrieved_triples([turtle])
        assert "http://www.w3.org/2000/01/rdf-schema#subClassOf" not in properties
        assert "http://example.org/ontology/education#DoctoralCandidate" in classes
        assert "http://example.org/ontology/education#Student" in classes

    def test_xsd_types_excluded_from_classes(self):
        """XSD type objects are NOT extracted as classes."""
        turtle = """
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix eduo: <http://example.org/ontology/education#> .
        eduo:Student eduo:name xsd:string .
        """
        classes, properties = parse_retrieved_triples([turtle])
        assert "http://www.w3.org/2001/XMLSchema#string" not in classes
        assert "http://example.org/ontology/education#Student" in classes
        assert "http://example.org/ontology/education#name" in properties

    def test_empty_list(self):
        """Empty list returns empty sets."""
        classes, properties = parse_retrieved_triples([])
        assert len(classes) == 0
        assert len(properties) == 0

    def test_malformed_turtle_handled(self):
        """Malformed Turtle doesn't crash, returns partial results."""
        classes, properties = parse_retrieved_triples(["NOT TURTLE"])
        assert isinstance(classes, set)
        assert isinstance(properties, set)

    def test_multiple_turtle_strings(self):
        """Multiple Turtle strings are combined."""
        t1 = """
        @prefix eduo: <http://example.org/ontology/education#> .
        eduo:Student eduo:takesCourse eduo:Course .
        """
        t2 = """
        @prefix eno: <http://example.org/ontology/energy#> .
        eno:Field eno:name eno:FieldName .
        """
        classes, properties = parse_retrieved_triples([t1, t2])
        assert "http://example.org/ontology/education#Student" in classes
        assert "http://example.org/ontology/energy#Field" in classes
        assert "http://example.org/ontology/education#takesCourse" in properties
        assert "http://example.org/ontology/energy#name" in properties


# ============================================================================
# Schema-Aware Metrics Tests
# ============================================================================


class TestSchemaAwareMetrics:
    """Test schema-aware retrieval metrics."""

    def _make_schema_gt(self, triples, paths=None):
        """Helper to build a RetrievalSchemaGT."""
        return RetrievalSchemaGT(
            triples=triples,
            paths=paths or [],
        )

    def _make_schema(self):
        """Helper to build a minimal SchemaGraph."""
        from src.validation.schema_graph import SchemaGraph
        return SchemaGraph(
            classes={
                "http://example.org/ontology/education#Student",
                "http://example.org/ontology/education#DoctoralCandidate",
                "http://example.org/ontology/education#Course",
            },
            properties={
                "http://example.org/ontology/education#takesCourse",
                "http://example.org/ontology/education#name",
            },
            subclass_of={
                "http://example.org/ontology/education#DoctoralCandidate": {
                    "http://example.org/ontology/education#Student"
                },
            },
            prefixes={
                "eduo": "http://example.org/ontology/education#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
            },
        )

    def test_perfect_match(self):
        """Retrieved triples exactly match GT."""
        ns = "http://example.org/ontology/education#"
        schema_gt = self._make_schema_gt([
            (f"{ns}Student", f"{ns}takesCourse", f"{ns}Course", RetrievalNecessity.REQUIRED),
        ], paths=[
            (f"{ns}Student", f"{ns}takesCourse", RetrievalNecessity.REQUIRED),
        ])
        schema = self._make_schema()
        turtle = [f"""
        @prefix eduo: <{ns}> .
        eduo:Student eduo:takesCourse eduo:Course .
        """]
        result = calculate_retrieval_metrics(turtle, schema, schema_gt)
        assert result.schema_triple_recall == 1.0
        assert result.schema_triple_precision == 1.0
        assert result.schema_triple_f1 == 1.0
        assert result.schema_path_coherence == 1.0

    def test_subclass_match(self):
        """Subclass triples match superclass GT."""
        ns = "http://example.org/ontology/education#"
        schema_gt = self._make_schema_gt([
            (f"{ns}Student", f"{ns}takesCourse", f"{ns}Course", RetrievalNecessity.REQUIRED),
        ], paths=[
            (f"{ns}Student", f"{ns}takesCourse", RetrievalNecessity.REQUIRED),
        ])
        schema = self._make_schema()
        # Agent retrieved DoctoralCandidate (subclass of Student)
        turtle = [f"""
        @prefix eduo: <{ns}> .
        eduo:DoctoralCandidate eduo:takesCourse eduo:Course .
        """]
        result = calculate_retrieval_metrics(turtle, schema, schema_gt)
        assert result.schema_triple_recall == 1.0
        assert result.schema_path_coherence == 1.0

    def test_acceptable_missed_no_penalty(self):
        """Missing ACCEPTABLE triples don't penalise recall."""
        ns = "http://example.org/ontology/education#"
        xsd = "http://www.w3.org/2001/XMLSchema#"
        schema_gt = self._make_schema_gt([
            (f"{ns}Student", f"{ns}takesCourse", f"{ns}Course", RetrievalNecessity.REQUIRED),
            (f"{ns}Student", f"{ns}name", f"{xsd}string", RetrievalNecessity.ACCEPTABLE),
        ], paths=[
            (f"{ns}Student", f"{ns}takesCourse", RetrievalNecessity.REQUIRED),
            (f"{ns}Student", f"{ns}name", RetrievalNecessity.ACCEPTABLE),
        ])
        schema = self._make_schema()
        # Only retrieved the required triple, not the acceptable one
        turtle = [f"""
        @prefix eduo: <{ns}> .
        eduo:Student eduo:takesCourse eduo:Course .
        """]
        result = calculate_retrieval_metrics(turtle, schema, schema_gt)
        # Recall should be 1.0: required triple found, acceptable not found = ignored
        assert result.schema_triple_recall == 1.0
        assert result.schema_path_coherence == 1.0

    def test_acceptable_found_counts_as_tp(self):
        """ACCEPTABLE triples that ARE found count as true positives."""
        ns = "http://example.org/ontology/education#"
        xsd = "http://www.w3.org/2001/XMLSchema#"
        schema_gt = self._make_schema_gt([
            (f"{ns}Student", f"{ns}takesCourse", f"{ns}Course", RetrievalNecessity.REQUIRED),
            (f"{ns}Student", f"{ns}name", f"{xsd}string", RetrievalNecessity.ACCEPTABLE),
        ], paths=[
            (f"{ns}Student", f"{ns}takesCourse", RetrievalNecessity.REQUIRED),
            (f"{ns}Student", f"{ns}name", RetrievalNecessity.ACCEPTABLE),
        ])
        schema = self._make_schema()
        # Both triples retrieved
        turtle = [f"""
        @prefix eduo: <{ns}> .
        @prefix xsd: <{xsd}> .
        eduo:Student eduo:takesCourse eduo:Course .
        eduo:Student eduo:name xsd:string .
        """]
        result = calculate_retrieval_metrics(turtle, schema, schema_gt)
        assert result.schema_triple_recall == 1.0
        assert result.schema_triple_precision == 1.0

    def test_empty_retrieval(self):
        """No retrieved triples gives zero metrics."""
        ns = "http://example.org/ontology/education#"
        schema_gt = self._make_schema_gt([
            (f"{ns}Student", f"{ns}takesCourse", f"{ns}Course", RetrievalNecessity.REQUIRED),
        ])
        schema = self._make_schema()
        result = calculate_retrieval_metrics([], schema, schema_gt)
        assert result.schema_triple_f1 == 0.0
        assert result.schema_path_coherence == 0.0

    def test_empty_gt(self):
        """Empty GT returns zero metrics."""
        schema_gt = self._make_schema_gt([])
        schema = self._make_schema()
        turtle = ["""
        @prefix eduo: <http://example.org/ontology/education#> .
        eduo:Student eduo:takesCourse eduo:Course .
        """]
        result = calculate_retrieval_metrics(turtle, schema, schema_gt)
        assert result.schema_triple_f1 == 0.0

    def test_to_dict_keys(self):
        """to_dict() returns all expected keys."""
        result = RetrievalMetricsResult()
        d = result.to_dict()
        expected_keys = {
            "schema_path_coherence", "schema_paths_matched", "schema_paths_total",
            "schema_paths_details", "schema_triple_precision", "schema_triple_recall",
            "schema_triple_f1", "schema_triples_matched", "schema_triples_total",
            "gt_schema_turtle",
        }
        assert set(d.keys()) == expected_keys
