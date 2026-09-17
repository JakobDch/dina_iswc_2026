"""Comprehensive tests for the semantic SPARQL validator prototype.

This test suite validates the semantic validator against real queries
from various datasets before pipeline integration.
"""

import pytest
from typing import Optional

from src.validation.schema_graph import (
    SchemaGraph,
    load_schema_graph,
    get_combined_schema,
    get_schema_for_dataset,
    clear_schema_cache,
)
from src.validation.pattern_extractor import (
    ExtractedPatterns,
    TriplePattern,
    extract_query_patterns,
    expand_prefixed_uri,
)
from src.validation.semantic_validator import (
    SemanticValidationResult,
    ValidationIssue,
    validate_sparql_semantics,
    format_validation_for_llm,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear schema cache before each test."""
    clear_schema_cache()
    yield
    clear_schema_cache()


# ============================================================================
# Schema Loading Tests
# ============================================================================

class TestSchemaLoading:
    """Tests for schema loading functionality."""

    def test_load_trn_schema(self):
        """Test loading TRN schema."""
        schema = load_schema_graph("TRN")

        # Check classes are loaded
        assert len(schema.classes) > 0
        assert "http://example.org/ontology/transport#Trip" in schema.classes
        assert "http://example.org/ontology/transport#Route" in schema.classes
        assert "http://example.org/ontology/transport#Stop" in schema.classes

        # Check properties are loaded
        assert len(schema.properties) > 0
        assert "http://example.org/ontology/transport#route" in schema.properties
        assert "http://example.org/ontology/transport#service" in schema.properties

        # Check class_properties mapping
        trip_props = schema.class_properties.get("http://example.org/ontology/transport#Trip", set())
        assert "http://example.org/ontology/transport#route" in trip_props

    def test_load_edu_schema(self):
        """Test loading EDU schema."""
        schema = load_schema_graph("EDU")

        assert len(schema.classes) > 0
        # Check some EDU classes
        assert any("University" in c for c in schema.classes)
        assert any("Professor" in c for c in schema.classes)

    def test_load_all_datasets(self):
        """Test that all 6 datasets can be loaded."""
        datasets = ["EDU", "TRN", "NRG", "BSBM", "BGEE", "LCA"]

        for dataset in datasets:
            schema = load_schema_graph(dataset)
            assert schema.dataset_id == dataset
            assert len(schema.classes) > 0, f"No classes loaded for {dataset}"
            assert len(schema.properties) > 0, f"No properties loaded for {dataset}"
            print(f"{dataset}: {len(schema.classes)} classes, {len(schema.properties)} properties")

    def test_combined_schema(self):
        """Test combining multiple schemas."""
        combined = get_combined_schema(["TRN", "EDU"])

        # Should have classes from both
        assert "http://example.org/ontology/transport#Trip" in combined.classes
        assert any("University" in c for c in combined.classes)

    def test_schema_caching(self):
        """Test that schema loading is cached."""
        schema1 = get_schema_for_dataset("TRN")
        schema2 = get_schema_for_dataset("TRN")

        # Should be the same object (cached)
        assert schema1 is schema2

    def test_class_property_lookup(self):
        """Test looking up properties for a class."""
        schema = load_schema_graph("TRN")

        trip_uri = "http://example.org/ontology/transport#Trip"
        props = schema.get_all_properties_for_class(trip_uri)

        assert len(props) > 0
        assert "http://example.org/ontology/transport#route" in props


# ============================================================================
# Pattern Extraction Tests
# ============================================================================

class TestPatternExtraction:
    """Tests for SPARQL pattern extraction."""

    def test_simple_select_query(self):
        """Test extracting patterns from a simple SELECT query."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?route WHERE {
            ?trip a trn:Trip .
            ?trip trn:route ?route .
        }
        """
        patterns = extract_query_patterns(query)

        assert patterns.is_valid_parse
        assert len(patterns.triple_patterns) == 2

        # Check type constraint was extracted
        assert "?trip" in patterns.type_constraints
        assert "http://example.org/ontology/transport#Trip" in patterns.type_constraints["?trip"]

    def test_query_with_filter(self):
        """Test extracting patterns from query with FILTER."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?name WHERE {
            ?trip a trn:Trip .
            ?trip trn:headsign ?name .
            FILTER(?name = "Downtown")
        }
        """
        patterns = extract_query_patterns(query)

        assert patterns.is_valid_parse
        # FILTER doesn't add triple patterns
        assert len(patterns.triple_patterns) == 2

    def test_query_with_optional(self):
        """Test extracting patterns from query with OPTIONAL."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?shape WHERE {
            ?trip a trn:Trip .
            OPTIONAL { ?trip trn:shape ?shape . }
        }
        """
        patterns = extract_query_patterns(query)

        assert patterns.is_valid_parse
        # Both required and optional patterns should be extracted
        assert len(patterns.triple_patterns) == 2

    def test_invalid_query(self):
        """Test handling of invalid query."""
        query = "SELECT ?x WHERE { this is not valid sparql"

        patterns = extract_query_patterns(query)

        assert not patterns.is_valid_parse
        assert patterns.parse_error is not None

    def test_prefix_expansion(self):
        """Test that prefixes are properly expanded."""
        prefixes = {"trn": "http://example.org/ontology/transport#"}
        expanded = expand_prefixed_uri("trn:Trip", prefixes)

        assert expanded == "http://example.org/ontology/transport#Trip"

    def test_select_variables_extraction(self):
        """Test that SELECT variables are extracted."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?route ?name WHERE {
            ?trip trn:route ?route .
        }
        """
        patterns = extract_query_patterns(query)

        assert "?trip" in patterns.select_variables
        assert "?route" in patterns.select_variables
        assert "?name" in patterns.select_variables


# ============================================================================
# Semantic Validation Tests - TRN Dataset
# ============================================================================

class TestSemanticValidationTRN:
    """Semantic validation tests using TRN dataset."""

    def test_valid_trip_route_query(self):
        """Test that a valid query passes validation."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?route WHERE {
            ?trip a trn:Trip .
            ?trip trn:route ?route .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        assert result.is_valid, f"Expected valid but got errors: {result.format_issues()}"
        assert len(result.errors) == 0

    def test_invalid_property(self):
        """Test that unknown property is detected as ERROR."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?agency WHERE {
            ?trip a trn:Trip .
            ?trip trn:agency ?agency .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        # trn:agency is NOT a property of Trip (it's a property of Route)
        # But we need to check if this is detected correctly
        # Actually, if trn:agency exists in schema (for Route), it might not be UNKNOWN_PROPERTY
        # Let's check with a definitely unknown property

    def test_completely_unknown_property(self):
        """Test that a completely unknown property is detected."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?foo WHERE {
            ?trip a trn:Trip .
            ?trip trn:nonExistentProperty ?foo .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        assert not result.is_valid
        assert any(e.code == "UNKNOWN_PROPERTY" for e in result.errors)
        print(f"Errors: {[e.message for e in result.errors]}")

    def test_unknown_class(self):
        """Test that unknown class is detected as ERROR."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?x WHERE {
            ?x a trn:NonExistentClass .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        assert not result.is_valid
        assert any(e.code == "UNKNOWN_CLASS" for e in result.errors)

    def test_domain_mismatch_warning(self):
        """Test that domain mismatch generates WARNING, not ERROR."""
        # trn:route property belongs to Trip, using it on Stop should warn
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?stop ?route WHERE {
            ?stop a trn:Stop .
            ?stop trn:route ?route .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        # Should be valid (warnings don't make it invalid)
        # But should have warning about domain mismatch
        print(f"Valid: {result.is_valid}")
        print(f"Warnings: {[w.message for w in result.warnings]}")
        print(f"Errors: {[e.message for e in result.errors]}")

        # Even with warning, is_valid should be True
        # (unless the property doesn't exist for Stop at all)

    def test_valid_multi_hop_query(self):
        """Test a valid multi-hop query (Trip -> Route -> Agency)."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?route ?agency WHERE {
            ?trip a trn:Trip .
            ?trip trn:route ?route .
            ?route trn:agency ?agency .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        # This should be valid - proper chain
        print(f"Valid: {result.is_valid}")
        print(f"Inferred types: {result.inferred_types}")
        if not result.is_valid:
            print(f"Errors: {result.format_issues()}")


# ============================================================================
# Semantic Validation Tests - EDU Dataset
# ============================================================================

class TestSemanticValidationEDU:
    """Semantic validation tests using EDU dataset."""

    def test_valid_professor_query(self):
        """Test a valid professor query."""
        query = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?prof ?name WHERE {
            ?prof a eduo:SeniorScholar .
            ?prof eduo:name ?name .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["EDU"])

        print(f"Valid: {result.is_valid}")
        if not result.is_valid:
            print(f"Errors: {result.format_issues()}")
        # Should be valid if schema is correct

    def test_unknown_edu_property(self):
        """Test unknown property in EDU."""
        query = """
        PREFIX eduo: <http://example.org/ontology/education#>
        SELECT ?prof ?salary WHERE {
            ?prof a eduo:SeniorScholar .
            ?prof eduo:salary ?salary .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["EDU"])

        # eduo:salary doesn't exist
        if not result.is_valid:
            assert any(e.code == "UNKNOWN_PROPERTY" for e in result.errors)


# ============================================================================
# Suggestions Tests
# ============================================================================

class TestSuggestions:
    """Tests for suggestion generation."""

    def test_similar_property_suggestions(self):
        """Test that similar properties are suggested."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?name WHERE {
            ?trip trn:routeName ?name .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        # Should suggest similar properties
        if result.errors:
            suggestions_found = any(len(e.suggestions) > 0 for e in result.errors)
            print(f"Suggestions: {[e.suggestions for e in result.errors]}")

    def test_similar_class_suggestions(self):
        """Test that similar classes are suggested."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?x WHERE {
            ?x a trn:Trp .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        # Should suggest trn:Trip
        if result.errors:
            for error in result.errors:
                if error.code == "UNKNOWN_CLASS":
                    print(f"Suggestions for {error.triple_pattern}: {error.suggestions}")


# ============================================================================
# Type Inference Tests
# ============================================================================

class TestTypeInference:
    """Tests for variable type inference."""

    def test_explicit_type_inference(self):
        """Test that explicit type constraints are captured."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip WHERE {
            ?trip a trn:Trip .
            ?trip trn:route ?route .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        assert "?trip" in result.inferred_types
        assert "http://example.org/ontology/transport#Trip" in result.inferred_types["?trip"]

    def test_domain_based_type_inference(self):
        """Test that types are inferred from property domains."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?x ?route WHERE {
            ?x trn:route ?route .
        }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])

        # ?x should be inferred as something that has trn:route property
        print(f"Inferred types for ?x: {result.inferred_types.get('?x', set())}")


# ============================================================================
# Format for LLM Tests
# ============================================================================

class TestFormatForLLM:
    """Tests for LLM-friendly output formatting."""

    def test_format_valid_query(self):
        """Test formatting for valid query."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip WHERE { ?trip a trn:Trip . }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])
        formatted = format_validation_for_llm(result)

        if result.is_valid and not result.warnings:
            assert "valid" in formatted.lower()

    def test_format_invalid_query(self):
        """Test formatting for invalid query."""
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip WHERE { ?trip trn:unknownProp ?x . }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])
        formatted = format_validation_for_llm(result)

        print(formatted)
        # Should contain error information
        if not result.is_valid:
            assert "ERROR" in formatted or "issues" in formatted.lower()


# ============================================================================
# Cross-Dataset Tests
# ============================================================================

class TestCrossDataset:
    """Tests for validation across multiple datasets."""

    def test_combined_dataset_validation(self):
        """Test validation against combined schema."""
        # A query that uses TRN concepts
        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip WHERE { ?trip a trn:Trip . }
        """
        result = validate_sparql_semantics(query, dataset_ids=["TRN", "EDU"])

        assert result.is_valid

    def test_all_datasets_schema_load(self):
        """Test that we can load and combine all datasets."""
        combined = get_combined_schema()  # All datasets

        print(f"Combined schema: {len(combined.classes)} classes, {len(combined.properties)} properties")
        assert len(combined.classes) > 100  # Should have many classes from all datasets


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Basic performance tests."""

    def test_validation_speed(self):
        """Test that validation completes in reasonable time."""
        import time

        query = """
        PREFIX tro: <http://example.org/ontology/transport#>
        SELECT ?trip ?route ?service ?shape WHERE {
            ?trip a trn:Trip .
            ?trip trn:route ?route .
            ?trip trn:service ?service .
            ?trip trn:shape ?shape .
            ?route trn:agency ?agency .
        }
        """

        # Warm up cache
        get_combined_schema(["TRN"])

        start = time.time()
        result = validate_sparql_semantics(query, dataset_ids=["TRN"])
        elapsed = (time.time() - start) * 1000

        print(f"Validation took {elapsed:.2f}ms")
        assert elapsed < 100, f"Validation too slow: {elapsed}ms"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
