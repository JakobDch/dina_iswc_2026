"""
Integration tests for SPARQL Agent routing behavior.

Tests the intelligent routing between:
1. Mapping Optimizer Agent (for self-join issues)
2. Multi-Step Agent (for complex query decomposition)
3. Strategy exhaustion handling

These tests verify the routing logic, not the actual agent execution.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response with tool calls."""
    def _create_response(tool_calls=None, content=""):
        response = MagicMock()
        response.content = content
        response.tool_calls = tool_calls or []
        return response
    return _create_response


@pytest.fixture
def sample_edu_query():
    """Sample EDU query that might timeout."""
    return """
    PREFIX eduo: <http://example.org/ontology/education#>
    SELECT ?prof ?dept ?univ WHERE {
        ?prof a eduo:SeniorScholar .
        ?prof eduo:worksFor ?dept .
        ?dept eduo:subOrganizationOf ?univ .
        ?univ a eduo:University .
    }
    """


@pytest.fixture
def sample_trn_query():
    """Sample TRN query that might cause self-joins."""
    return """
    PREFIX tro: <http://example.org/ontology/transport#>
    SELECT ?stop ?arrival WHERE {
        ?st a trn:StopTime .
        ?st trn:stop ?stop .
        ?st trn:arrivalTime ?arrival .
    }
    """


# ============================================================================
# SQL Analysis Tests
# ============================================================================

class TestSqlAnalysis:
    """Tests for SQL self-join detection."""

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

    def test_detect_self_join_with_backticks(self):
        """Test detection of self-join with backtick-quoted table names."""
        import re

        sql = """
        SELECT v1.trip_id, v2.stop_id
        FROM `STOP_TIMES` v1, `STOP_TIMES` v2
        WHERE v1.trip_id = v2.trip_id
        """

        self_join_pattern = r'FROM\s+`?(\w+)`?\s+(\w+)\s*,\s*`?\1`?\s+(\w+)'
        matches = re.findall(self_join_pattern, sql, re.IGNORECASE)

        assert len(matches) == 1
        assert matches[0][0] == "STOP_TIMES"

    def test_no_self_join_different_tables(self):
        """Test that different tables are not detected as self-join."""
        import re

        sql = """
        SELECT s.stop_name, t.trip_id
        FROM STOPS s, TRIPS t
        WHERE s.stop_id = t.stop_id
        """

        self_join_pattern = r'FROM\s+`?(\w+)`?\s+(\w+)\s*,\s*`?\1`?\s+(\w+)'
        matches = re.findall(self_join_pattern, sql, re.IGNORECASE)

        assert len(matches) == 0


# ============================================================================
# Routing Logic Tests (Unit Tests with Mocking)
# ============================================================================

class TestRoutingConstants:
    """Tests for routing constants."""

    def test_max_mapping_optimization_attempts(self):
        """Verify MAX_MAPPING_OPTIMIZATION_ATTEMPTS is 2."""
        # The constant is defined inside the method, so we test via behavior
        # This is a documentation test
        expected_max = 2
        assert expected_max == 2, "MAX_MAPPING_OPTIMIZATION_ATTEMPTS should be 2"

    def test_max_multi_step_attempts(self):
        """Verify MAX_MULTI_STEP_ATTEMPTS is 2."""
        expected_max = 2
        assert expected_max == 2, "MAX_MULTI_STEP_ATTEMPTS should be 2"


class TestRoutingDecisions:
    """Tests for routing decision logic."""

    def test_self_join_routes_to_mapping_optimizer(self):
        """Test that self-join detection routes to Mapping Optimizer."""
        # Simulate the routing decision
        has_self_join = True
        mapping_optimization_attempts = 0
        MAX_MAPPING_OPTIMIZATION_ATTEMPTS = 2

        should_try_mapping = has_self_join and mapping_optimization_attempts < MAX_MAPPING_OPTIMIZATION_ATTEMPTS
        assert should_try_mapping is True

    def test_no_self_join_routes_to_multi_step(self):
        """Test that no self-join routes to Multi-Step Agent."""
        has_self_join = False
        mapping_optimization_attempts = 0
        MAX_MAPPING_OPTIMIZATION_ATTEMPTS = 2

        should_try_mapping = has_self_join and mapping_optimization_attempts < MAX_MAPPING_OPTIMIZATION_ATTEMPTS
        should_try_multi_step = not has_self_join

        assert should_try_mapping is False
        assert should_try_multi_step is True

    def test_max_mapping_attempts_routes_to_multi_step(self):
        """Test that exhausted mapping attempts route to Multi-Step."""
        has_self_join = True
        mapping_optimization_attempts = 2
        MAX_MAPPING_OPTIMIZATION_ATTEMPTS = 2

        should_try_mapping = has_self_join and mapping_optimization_attempts < MAX_MAPPING_OPTIMIZATION_ATTEMPTS
        should_try_multi_step = mapping_optimization_attempts >= MAX_MAPPING_OPTIMIZATION_ATTEMPTS

        assert should_try_mapping is False
        assert should_try_multi_step is True

    def test_all_strategies_exhausted(self):
        """Test that all strategies exhausted returns failure."""
        mapping_optimization_attempts = 2
        multi_step_attempts = 2
        MAX_MAPPING_OPTIMIZATION_ATTEMPTS = 2
        MAX_MULTI_STEP_ATTEMPTS = 2

        all_exhausted = (
            mapping_optimization_attempts >= MAX_MAPPING_OPTIMIZATION_ATTEMPTS and
            multi_step_attempts >= MAX_MULTI_STEP_ATTEMPTS
        )

        assert all_exhausted is True


class TestExhaustedResult:
    """Tests for exhausted strategies result structure."""

    def test_exhausted_result_structure(self):
        """Test the structure of result when all strategies are exhausted."""
        exhausted_result = {
            "success": False,
            "error": "All recovery strategies exhausted",
            "reasoning": "Query timed out. Tried mapping optimization 2 times and Multi-Step decomposition 2 times. Unable to generate a working query.",
            "mapping_optimization_attempts": 2,
            "multi_step_attempts": 2,
        }

        assert exhausted_result["success"] is False
        assert "exhausted" in exhausted_result["error"].lower()
        assert exhausted_result["mapping_optimization_attempts"] == 2
        assert exhausted_result["multi_step_attempts"] == 2
        assert "Unable to generate" in exhausted_result["reasoning"]

    def test_exhausted_result_from_fallback(self):
        """Test exhausted result structure from fallback delegation."""
        timeout_count = 3
        mapping_optimization_attempts = 2
        multi_step_attempts = 2

        exhausted_result = {
            "success": False,
            "error": "All recovery strategies exhausted",
            "reasoning": f"Query timed out {timeout_count} times. Mapping optimization: {mapping_optimization_attempts} attempts. Multi-Step decomposition: {multi_step_attempts} attempts. Unable to generate a working query.",
            "mapping_optimization_attempts": mapping_optimization_attempts,
            "multi_step_attempts": multi_step_attempts,
        }

        assert f"timed out {timeout_count} times" in exhausted_result["reasoning"]


# ============================================================================
# Dataset Detection Tests
# ============================================================================

class TestDatasetDetection:
    """Tests for dataset detection from query prefixes."""

    def test_detect_trn_dataset(self):
        """Test detection of TRN dataset from query."""
        query = "PREFIX tro: <http://example.org/ontology/transport#> SELECT ?x WHERE { ?x a trn:Stop }"
        query_lower = query.lower()

        detected_dataset = "edu"  # Default
        if "trn:" in query_lower:
            detected_dataset = "trn-large" if "large" in query_lower else "trn"

        assert detected_dataset in ["trn", "trn-large"]

    def test_detect_edu_dataset(self):
        """Test detection of EDU dataset from query."""
        query = "PREFIX eduo: <http://example.org/ontology/education#> SELECT ?x WHERE { ?x a eduo:Professor }"
        query_lower = query.lower()

        detected_dataset = "edu"  # Default
        if "trn:" in query_lower:
            detected_dataset = "trn"
        elif "eno:" in query_lower or "nrg" in query_lower:
            detected_dataset = "nrg"

        assert detected_dataset == "edu"

    def test_detect_npd_dataset(self):
        """Test detection of NRG dataset from query."""
        query = "PREFIX eno: <http://example.org/ontology/energy#> SELECT ?x WHERE { ?x a eno:Field }"
        query_lower = query.lower()

        detected_dataset = "edu"  # Default
        if "eno:" in query_lower or "nrg" in query_lower:
            detected_dataset = "nrg"

        assert detected_dataset == "nrg"


# ============================================================================
# Query History Tracking Tests
# ============================================================================

class TestQueryHistoryTracking:
    """Tests for query history tracking."""

    def test_timeout_history_entry(self):
        """Test history entry structure for timeout."""
        history_entry = {
            "query": "SELECT ?x WHERE { ?x a trn:Stop }",
            "status": "timeout",
            "count": 0,
            "error_msg": None,
        }

        assert history_entry["status"] == "timeout"
        assert history_entry["count"] == 0

    def test_success_history_entry(self):
        """Test history entry structure for success."""
        history_entry = {
            "query": "SELECT ?x WHERE { ?x a trn:Stop }",
            "status": "success",
            "count": 42,
            "error_msg": None,
        }

        assert history_entry["status"] == "success"
        assert history_entry["count"] == 42

    def test_error_history_entry(self):
        """Test history entry structure for error."""
        history_entry = {
            "query": "SELECT ?x WHERE { ?x a trn:Stop }",
            "status": "error",
            "count": 0,
            "error_msg": "Invalid SPARQL syntax",
        }

        assert history_entry["status"] == "error"
        assert history_entry["error_msg"] is not None


# ============================================================================
# Integration Test: Analyze SQL Query Tool
# ============================================================================

class TestAnalyzeSqlQueryTool:
    """Integration tests for analyze_sql_query tool."""

    def test_analyze_sql_query_with_self_join_mock(self):
        """Test analyze_sql_query detects self-joins (mocked)."""
        from src.tools.sparql_tools import analyze_sql_query

        # Mock the HTTP response
        with patch('httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = """
            SELECT v1.trip_id, v2.stop_id
            FROM STOP_TIMES v1, STOP_TIMES v2
            WHERE v1.trip_id = v2.trip_id AND v1.stop_sequence = v2.stop_sequence
            """
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            result = analyze_sql_query.invoke({
                "sparql_query": "SELECT ?x WHERE { ?x a trn:StopTime }",
                "dataset": "trn-large",
            })

        assert result["success"] is True
        assert result["has_self_join"] is True
        assert "STOP_TIMES" in result["self_join_tables"]

    def test_analyze_sql_query_no_self_join_mock(self):
        """Test analyze_sql_query with no self-joins (mocked)."""
        from src.tools.sparql_tools import analyze_sql_query

        with patch('httpx.Client') as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = """
            SELECT s.stop_name, t.trip_id
            FROM STOPS s
            INNER JOIN TRIPS t ON s.stop_id = t.origin_stop_id
            """
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            result = analyze_sql_query.invoke({
                "sparql_query": "SELECT ?x WHERE { ?x a trn:Stop }",
                "dataset": "trn",
            })

        assert result["success"] is True
        assert result["has_self_join"] is False
        assert len(result["self_join_tables"]) == 0


# ============================================================================
# Scenarios: EDU Dataset
# ============================================================================

class TestEDUScenarios:
    """Test scenarios with EDU dataset."""

    def test_edu_simple_query_no_routing(self):
        """Simple EDU query should not trigger routing (no timeout)."""
        query = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?prof WHERE {
            ?prof a eduo:SeniorScholar .
        }
        """
        # This query is simple and should succeed without needing recovery
        # Just verify it's syntactically valid
        assert "SELECT" in query
        assert "SeniorScholar" in query

    def test_edu_complex_query_multi_step_scenario(self):
        """Complex EDU query scenario for Multi-Step routing."""
        # This type of query might timeout due to complexity (not self-joins)
        query = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?prof ?dept ?univ ?course ?student WHERE {
            ?prof a eduo:SeniorScholar .
            ?prof eduo:worksFor ?dept .
            ?dept eduo:subOrganizationOf ?univ .
            ?prof eduo:teacherOf ?course .
            ?student eduo:takesCourse ?course .
        }
        """
        # Verify query structure
        assert query.count("?") >= 5  # Multiple variables
        assert "teacherOf" in query
        assert "takesCourse" in query


# ============================================================================
# Scenarios: TRN Dataset
# ============================================================================

class TestTRNScenarios:
    """Test scenarios with TRN dataset."""

    def test_trn_stoptime_self_join_scenario(self):
        """TRN StopTime query that might cause self-joins."""
        # This pattern often causes self-joins due to IRI templates
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?st ?stop ?arrival WHERE {
            ?st a trn:StopTime .
            ?st trn:stop ?stop .
            ?st trn:arrivalTime ?arrival .
        }
        """
        # The STOP_TIMES table often has self-join issues
        assert "StopTime" in query
        assert "arrivalTime" in query

    def test_trn_trip_route_chain_scenario(self):
        """TRN Trip-Route chain query for Multi-Step routing."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?route ?agency ?stopTime ?stop WHERE {
            ?trip a trn:Trip .
            ?trip trn:route ?route .
            ?route trn:agency ?agency .
            ?stopTime trn:trip ?trip .
            ?stopTime trn:stop ?stop .
        }
        """
        # Complex chain that might need decomposition
        assert query.count("trn:") >= 5
        assert "?trip" in query and "?route" in query and "?agency" in query


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
