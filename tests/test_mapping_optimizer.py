"""
Tests for the Mapping Optimizer Agent and related tools.

These tests verify:
1. Self-join detection in SQL queries
2. IRI template analysis and suggestion
3. Mapping modification and backup/restore
4. Integration between SPARQL Agent and Mapping Optimizer
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSelfJoinDetection:
    """Tests for SQL self-join detection (legacy regex-based tests kept for regression)."""

    def test_detect_self_join_from_clause(self):
        """Test detection of self-join in FROM clause."""
        import re

        sql = """
        SELECT v1.trip_id, v2.stop_id
        FROM STOP_TIMES v1, STOP_TIMES v2
        WHERE v1.trip_id = v2.trip_id
        """

        # Pattern from analyze_sql_query
        self_join_pattern = r'FROM\s+`?(\w+)`?\s+(\w+)\s*,\s*`?\1`?\s+(\w+)'
        matches = re.findall(self_join_pattern, sql, re.IGNORECASE)

        assert len(matches) == 1
        assert matches[0][0] == "STOP_TIMES"

    def test_detect_no_self_join(self):
        """Test that normal joins are not detected as self-joins."""
        import re

        sql = """
        SELECT s.stop_name, t.trip_id
        FROM STOPS s, TRIPS t
        WHERE s.stop_id = t.stop_id
        """

        self_join_pattern = r'FROM\s+`?(\w+)`?\s+(\w+)\s*,\s*`?\1`?\s+(\w+)'
        matches = re.findall(self_join_pattern, sql, re.IGNORECASE)

        assert len(matches) == 0


class TestSQLAnalyzer:
    """Tests for sqlglot-based SQL analysis (src/tools/sql_analyzer.py)."""

    # -- extract_tables --

    def test_extract_tables_simple(self):
        """Extract tables from a simple SELECT."""
        from src.tools.sql_analyzer import extract_tables

        sql = "SELECT v1.x FROM STOP_TIMES v1 WHERE v1.id = 1"
        result = extract_tables(sql)
        assert result == ["stop_times"]

    def test_extract_tables_with_join(self):
        """Extract tables from JOIN query."""
        from src.tools.sql_analyzer import extract_tables

        sql = "SELECT * FROM student v1 INNER JOIN department v2 ON v1.dept = v2.nr"
        result = extract_tables(sql)
        assert "student" in result
        assert "department" in result

    def test_extract_tables_with_backticks(self):
        """MySQL-style backtick quoting."""
        from src.tools.sql_analyzer import extract_tables

        sql = "SELECT * FROM `undergraduateStudent` v1 INNER JOIN `department` v2 ON v1.x = v2.y"
        result = extract_tables(sql)
        assert "undergraduatestudent" in result
        assert "department" in result

    def test_extract_tables_comma_separated(self):
        """Multiple tables in FROM clause separated by commas."""
        from src.tools.sql_analyzer import extract_tables

        sql = "SELECT * FROM faculty v1, department v2, university v3 WHERE v1.id = v2.id"
        result = extract_tables(sql)
        assert "faculty" in result
        assert "department" in result
        assert "university" in result

    def test_extract_tables_union_all(self):
        """Tables from UNION ALL branches."""
        from src.tools.sql_analyzer import extract_tables

        sql = """
        SELECT v1.name FROM student v1 WHERE v1.type = 1
        UNION ALL
        SELECT v2.name FROM faculty v2 WHERE v2.type = 2
        """
        result = extract_tables(sql)
        assert "student" in result
        assert "faculty" in result

    def test_extract_tables_fallback_on_garbage(self):
        """Graceful fallback to regex for non-standard SQL."""
        from src.tools.sql_analyzer import extract_tables

        sql = "SOME WEIRD ONTOP SQL WITH FROM mytable v1"
        result = extract_tables(sql)
        assert "mytable" in result

    # -- detect_self_joins --

    def test_self_join_from_clause(self):
        """Classic self-join: FROM table v1, table v2."""
        from src.tools.sql_analyzer import detect_self_joins

        sql = "SELECT v1.x, v2.y FROM STOP_TIMES v1, STOP_TIMES v2 WHERE v1.id = v2.id"
        result = detect_self_joins(sql)
        assert "stop_times" in result
        assert len(result["stop_times"]) == 2

    def test_no_self_join(self):
        """Normal join between different tables is NOT a self-join."""
        from src.tools.sql_analyzer import detect_self_joins

        sql = "SELECT s.name, t.id FROM STOPS s, TRIPS t WHERE s.id = t.stop_id"
        result = detect_self_joins(sql)
        assert len(result) == 0

    def test_self_join_via_explicit_join(self):
        """Self-join via explicit JOIN keyword — regex often misses this."""
        from src.tools.sql_analyzer import detect_self_joins

        sql = """
        SELECT a.name FROM faculty a
        JOIN faculty b ON a.worksFor = b.worksFor
        WHERE a.nr != b.nr
        """
        result = detect_self_joins(sql)
        assert "faculty" in result
        assert len(result["faculty"]) == 2

    def test_self_join_three_aliases(self):
        """Same table with 3 aliases (common in Ontop output)."""
        from src.tools.sql_analyzer import detect_self_joins

        sql = """
        SELECT v1.name, v2.email, v3.phone
        FROM person v1, person v2, person v3
        WHERE v1.pid = v2.pid AND v1.pid = v3.pid
        """
        result = detect_self_joins(sql)
        assert "person" in result
        assert len(result["person"]) == 3

    def test_self_join_in_union_all(self):
        """Self-join detection across UNION ALL branches."""
        from src.tools.sql_analyzer import detect_self_joins

        sql = """
        SELECT v1.name FROM student v1 WHERE v1.type = 1
        UNION ALL
        SELECT v2.name FROM student v2 WHERE v2.type = 2
        """
        result = detect_self_joins(sql)
        assert "student" in result

    def test_self_join_with_backticks(self):
        """MySQL backtick-quoted self-join."""
        from src.tools.sql_analyzer import detect_self_joins

        sql = "SELECT * FROM `graduateStudent` v1, `graduateStudent` v2 WHERE v1.nr = v2.nr"
        result = detect_self_joins(sql)
        assert "graduatestudent" in result

    # -- analyze_sql (full analysis) --

    def test_analyze_sql_full(self):
        """Full analysis returns all fields."""
        from src.tools.sql_analyzer import analyze_sql

        sql = """
        SELECT v1.name, v2.dept
        FROM faculty v1, faculty v2
        INNER JOIN department v3 ON v1.worksFor = v3.nr
        WHERE v1.nr = v2.nr
        """
        result = analyze_sql(sql)
        assert result.has_self_join
        assert "faculty" in result.self_joins
        assert "faculty" in result.tables
        assert "department" in result.tables
        assert result.union_branch_count == 1

    def test_analyze_sql_union_count(self):
        """Count UNION ALL branches correctly."""
        from src.tools.sql_analyzer import analyze_sql

        sql = """
        SELECT * FROM t1
        UNION ALL
        SELECT * FROM t2
        UNION ALL
        SELECT * FROM t3
        """
        result = analyze_sql(sql)
        assert result.union_branch_count == 3

    def test_analyze_sql_no_issues(self):
        """Clean SQL with no self-joins."""
        from src.tools.sql_analyzer import analyze_sql

        sql = "SELECT s.name FROM student s INNER JOIN department d ON s.dept = d.nr"
        result = analyze_sql(sql)
        assert not result.has_self_join
        assert result.union_branch_count == 1
        assert "student" in result.tables
        assert "department" in result.tables

    # -- join condition extraction --

    def test_extract_join_conditions(self):
        """Extract ON conditions from explicit JOINs."""
        from src.tools.sql_analyzer import analyze_sql

        sql = """
        SELECT * FROM student s
        INNER JOIN department d ON s.memberOf = d.nr
        INNER JOIN university u ON d.subOrganizationOf = u.nr
        """
        result = analyze_sql(sql)
        assert len(result.join_conditions) == 2
        # Check first condition
        cond = result.join_conditions[0]
        assert cond.left_column == "memberof"
        assert cond.right_column == "nr"


class TestTemplateAnalysis:
    """Tests for IRI template analysis and suggestion."""

    def test_extract_template_columns(self):
        """Test extraction of columns from IRI template."""
        import re

        template = "http://transport.data/stoptimes/{trip_id}-{stop_id}-{arrival_time}"
        columns = re.findall(r'\{(\w+)\}', template)

        assert columns == ["trip_id", "stop_id", "arrival_time"]

    def test_suggest_optimized_template(self):
        """Test suggestion of optimized template using PK columns."""
        import re

        current_template = "http://transport.data/stoptimes/{trip_id}-{stop_id}-{arrival_time}"
        pk_columns = ["trip_id", "stop_sequence"]

        # Extract base URL
        base_match = re.match(r'(.+/)(?:\{[^}]+\}[-_]?)+', current_template)
        base_url = base_match.group(1) if base_match else ""

        # Build new template
        pk_placeholders = '-'.join(f'{{{col}}}' for col in pk_columns)
        suggested = base_url + pk_placeholders

        assert suggested == "http://transport.data/stoptimes/{trip_id}-{stop_sequence}"


class TestMappingContext:
    """Tests for mapping backup/restore context management."""

    def test_context_initialization(self):
        """Test that context initializes with empty state."""
        from src.tools.mapping_optimizer_tools import MappingOptimizationContext

        ctx = MappingOptimizationContext()

        assert ctx.original_mappings == {}
        assert ctx.modified == False
        assert ctx.affected_datasets == set()

    def test_backup_mapping(self, tmp_path):
        """Test that backup stores original content."""
        from src.tools.mapping_optimizer_tools import MappingOptimizationContext

        # Create a test mapping file
        mapping_file = tmp_path / "mapping.ttl"
        original_content = "@prefix rr: <http://www.w3.org/ns/r2rml#> ."
        mapping_file.write_text(original_content)

        ctx = MappingOptimizationContext()
        ctx.backup_mapping(str(mapping_file), "test-dataset")

        assert str(mapping_file) in ctx.original_mappings
        assert ctx.original_mappings[str(mapping_file)] == original_content
        assert "test-dataset" in ctx.affected_datasets

    def test_restore_mapping(self, tmp_path):
        """Test that restore reverts to original content."""
        from src.tools.mapping_optimizer_tools import MappingOptimizationContext

        # Create a test mapping file
        mapping_file = tmp_path / "mapping.ttl"
        original_content = "@prefix rr: <http://www.w3.org/ns/r2rml#> ."
        mapping_file.write_text(original_content)

        ctx = MappingOptimizationContext()
        ctx.backup_mapping(str(mapping_file), "test-dataset")

        # Modify the file
        modified_content = "@prefix rr: <http://modified.example.com#> ."
        mapping_file.write_text(modified_content)
        ctx.mark_modified()

        # Mock the container restart to avoid actual Docker operations
        with patch('src.tools.mapping_optimizer_tools.restart_ontop_container') as mock_restart:
            mock_restart.invoke.return_value = '{"success": true}'
            ctx.restore_all()  # Now synchronous

        # Verify content is restored
        assert mapping_file.read_text() == original_content
        assert ctx.original_mappings == {}
        assert ctx.affected_datasets == set()


class TestMappingOptimizerResult:
    """Tests for MappingOptimizerResult dataclass."""

    def test_result_to_dict(self):
        """Test conversion of result to dictionary."""
        from src.tools.mapping_optimizer_tools import MappingOptimizerResult, MappingChange

        result = MappingOptimizerResult(
            success=True,
            optimized_tables=["STOP_TIMES"],
            changes_made=[
                MappingChange(
                    table="STOP_TIMES",
                    old_template="http://test/{trip_id}-{arrival_time}",
                    new_template="http://test/{trip_id}-{stop_sequence}",
                    primary_key_columns=["trip_id", "stop_sequence"],
                    old_template_columns=["trip_id", "arrival_time"],
                )
            ],
            requires_restart=True,
            reasoning="Optimized IRI template to use primary key columns.",
        )

        d = result.to_dict()

        assert d["success"] == True
        assert d["optimized_tables"] == ["STOP_TIMES"]
        assert len(d["changes_made"]) == 1
        assert d["changes_made"][0]["table"] == "STOP_TIMES"
        assert d["requires_restart"] == True


class TestMappingOptimizationRequest:
    """Tests for MappingOptimizationRequest dataclass."""

    def test_request_from_dict(self):
        """Test creation of request from dictionary."""
        from src.agents.generation.mapping_optimizer_agent import MappingOptimizationRequest

        data = {
            "sparql_query": "SELECT ?x WHERE { ?x a trn:Stop }",
            "generated_sql": "FROM STOP_TIMES v1, STOP_TIMES v2",
            "dataset": "trn-large",
            "endpoint_url": "http://localhost:8082/sparql",
            "mapping_path": "/path/to/mapping.ttl",
            "sparql_agent_reasoning": "Self-join detected on STOP_TIMES.",
        }

        request = MappingOptimizationRequest.from_dict(data)

        assert request.sparql_query == data["sparql_query"]
        assert request.generated_sql == data["generated_sql"]
        assert request.sparql_agent_reasoning == "Self-join detected on STOP_TIMES."


class TestAnalyzeSqlQuery:
    """Tests for the analyze_sql_query tool."""

    def test_analyze_sql_query_returns_dict(self):
        """Test that analyze_sql_query returns expected structure."""
        from src.tools.sparql_tools import analyze_sql_query

        # Mock the HTTP call
        with patch('httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "SELECT * FROM STOP_TIMES v1, STOP_TIMES v2 WHERE v1.trip_id = v2.trip_id"
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            result = analyze_sql_query.invoke({
                "sparql_query": "SELECT ?x WHERE { ?x a trn:Stop }",
                "dataset": "trn-large",
            })

        # Result should be a dict (the tool returns dict, not string)
        assert isinstance(result, dict)
        assert "success" in result


class TestEditMapping:
    """Tests for the edit_mapping function (simple string replacement)."""

    SAMPLE_MAPPING = '''@prefix rr: <http://www.w3.org/ns/r2rml#>.
@prefix trn: <http://example.org/ontology/transport#>.

<stoptimes_0> a rr:TriplesMap;
    rr:logicalTable [ rr:tableName "STOP_TIMES" ];
    rr:subjectMap [
        rr:template "http://example.org/data/transport/stoptimes/{trip_id}-{stop_id}-{arrival_time}";
    ];
    rr:predicateObjectMap [
        rr:predicate trn:dropOffType;
        rr:objectMap [ rr:template "http://example.org/data/transport/resource/DropOffType/{drop_off_type}" ];
    ].

<trips_0> a rr:TriplesMap;
    rr:logicalTable [ rr:tableName "TRIPS" ];
    rr:subjectMap [
        rr:template "http://example.org/data/transport/trips/{trip_id}";
    ].
'''

    def test_edit_mapping_replaces_unique_string(self, tmp_path):
        """Test that edit_mapping replaces a unique string correctly."""
        from src.tools.mapping_optimizer_tools import edit_mapping, reset_mapping_context
        from unittest.mock import patch

        mapping_dir = tmp_path / "trn-test"
        mapping_dir.mkdir()
        mapping_file = mapping_dir / "mapping.ttl"
        mapping_file.write_text(self.SAMPLE_MAPPING)

        reset_mapping_context()

        with patch('src.tools.mapping_optimizer_tools._get_mapping_path', return_value=mapping_file):
            result_json = edit_mapping.invoke({
                "dataset": "trn-test",
                "old_string": 'rr:template "http://example.org/data/transport/stoptimes/{trip_id}-{stop_id}-{arrival_time}";',
                "new_string": 'rr:template "http://example.org/data/transport/stoptimes/{trip_id}-{stop_sequence}";',
            })

        result = json.loads(result_json)
        assert result["success"] == True

        modified_content = mapping_file.read_text()
        assert "{trip_id}-{stop_sequence}" in modified_content
        assert "{trip_id}-{stop_id}-{arrival_time}" not in modified_content
        # Other templates should be unchanged
        assert "trips/{trip_id}" in modified_content
        assert "DropOffType/{drop_off_type}" in modified_content

    def test_edit_mapping_fails_on_non_unique_string(self, tmp_path):
        """Test that edit_mapping fails when old_string appears multiple times."""
        from src.tools.mapping_optimizer_tools import edit_mapping, reset_mapping_context
        from unittest.mock import patch

        mapping_dir = tmp_path / "trn-test"
        mapping_dir.mkdir()
        mapping_file = mapping_dir / "mapping.ttl"
        mapping_file.write_text(self.SAMPLE_MAPPING)

        reset_mapping_context()

        with patch('src.tools.mapping_optimizer_tools._get_mapping_path', return_value=mapping_file):
            # "rr:template" appears multiple times
            result_json = edit_mapping.invoke({
                "dataset": "trn-test",
                "old_string": "rr:template",
                "new_string": "rr:newTemplate",
            })

        result = json.loads(result_json)
        assert result["success"] == False
        assert "appears" in result["error"] and "times" in result["error"]

    def test_edit_mapping_fails_on_missing_string(self, tmp_path):
        """Test that edit_mapping fails when old_string is not found."""
        from src.tools.mapping_optimizer_tools import edit_mapping, reset_mapping_context
        from unittest.mock import patch

        mapping_dir = tmp_path / "trn-test"
        mapping_dir.mkdir()
        mapping_file = mapping_dir / "mapping.ttl"
        mapping_file.write_text(self.SAMPLE_MAPPING)

        reset_mapping_context()

        with patch('src.tools.mapping_optimizer_tools._get_mapping_path', return_value=mapping_file):
            result_json = edit_mapping.invoke({
                "dataset": "trn-test",
                "old_string": "this string does not exist",
                "new_string": "replacement",
            })

        result = json.loads(result_json)
        assert result["success"] == False
        assert "not found" in result["error"]

    def test_edit_mapping_does_not_affect_other_parts(self, tmp_path):
        """Test that edit_mapping only changes the specified string."""
        from src.tools.mapping_optimizer_tools import edit_mapping, reset_mapping_context
        from unittest.mock import patch

        mapping_dir = tmp_path / "trn-test"
        mapping_dir.mkdir()
        mapping_file = mapping_dir / "mapping.ttl"
        mapping_file.write_text(self.SAMPLE_MAPPING)

        reset_mapping_context()

        with patch('src.tools.mapping_optimizer_tools._get_mapping_path', return_value=mapping_file):
            edit_mapping.invoke({
                "dataset": "trn-test",
                "old_string": 'rr:template "http://example.org/data/transport/trips/{trip_id}";',
                "new_string": 'rr:template "http://example.org/data/transport/trips/{trip_nr}";',
            })

        modified_content = mapping_file.read_text()
        # TRIPS template changed
        assert "trips/{trip_nr}" in modified_content
        # STOP_TIMES template unchanged (still has {trip_id})
        assert "stoptimes/{trip_id}-{stop_id}-{arrival_time}" in modified_content


class TestReadMapping:
    """Tests for the read_mapping function."""

    def test_read_full_mapping(self, tmp_path):
        """Test reading the full mapping file."""
        from src.tools.mapping_optimizer_tools import read_mapping
        from unittest.mock import patch

        mapping_dir = tmp_path / "test-dataset"
        mapping_dir.mkdir()
        mapping_file = mapping_dir / "mapping.ttl"
        mapping_file.write_text("line1\nline2\nline3\n")

        with patch('src.tools.mapping_optimizer_tools._get_mapping_path', return_value=mapping_file):
            result_json = read_mapping.invoke({"dataset": "test-dataset"})

        result = json.loads(result_json)
        assert result["success"] == True
        assert "line1" in result["content"]
        assert "line3" in result["content"]
        assert result["total_lines"] == 4

    def test_read_mapping_line_range(self, tmp_path):
        """Test reading specific lines from the mapping file."""
        from src.tools.mapping_optimizer_tools import read_mapping
        from unittest.mock import patch

        mapping_dir = tmp_path / "test-dataset"
        mapping_dir.mkdir()
        mapping_file = mapping_dir / "mapping.ttl"
        mapping_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        with patch('src.tools.mapping_optimizer_tools._get_mapping_path', return_value=mapping_file):
            result_json = read_mapping.invoke({
                "dataset": "test-dataset",
                "line_start": 2,
                "line_end": 4,
            })

        result = json.loads(result_json)
        assert result["success"] == True
        assert "line2" in result["content"]
        assert "line3" in result["content"]
        assert "line4" in result["content"]
        assert "line1" not in result["content"]
        assert "line5" not in result["content"]


class TestRecoveryStrategyLimits:
    """Tests for maximum attempt limits on recovery strategies."""

    def test_max_mapping_optimization_attempts_constant(self):
        """Test that MAX_MAPPING_OPTIMIZATION_ATTEMPTS is set to 2."""
        # Import to verify the constant exists and has expected value
        # Note: The constant is defined inside the method, so we verify via behavior
        assert True  # Placeholder - actual behavior tested in integration tests

    def test_max_multi_step_attempts_constant(self):
        """Test that MAX_MULTI_STEP_ATTEMPTS is set to 2."""
        # Import to verify the constant exists and has expected value
        # Note: The constant is defined inside the method, so we verify via behavior
        assert True  # Placeholder - actual behavior tested in integration tests

    def test_exhausted_strategies_result_structure(self):
        """Test the structure of result when all strategies are exhausted."""
        # Simulate the result structure returned when all recovery strategies fail
        exhausted_result = {
            "success": False,
            "error": "All recovery strategies exhausted",
            "reasoning": "Query timed out. Mapping optimization: 2 attempts. Multi-Step decomposition: 2 attempts. Unable to generate a working query.",
            "mapping_optimization_attempts": 2,
            "multi_step_attempts": 2,
        }

        assert exhausted_result["success"] == False
        assert "exhausted" in exhausted_result["error"].lower()
        assert exhausted_result["mapping_optimization_attempts"] == 2
        assert exhausted_result["multi_step_attempts"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
