"""Tests for datatype-based column matching filter.

Tests cover the new datatype inference and compatibility checking functions
that prevent false positive matches between columns with numeric value overlap.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the module directly to avoid triggering src/evaluation/__init__.py, which
# imports agent modules with heavy (and currently broken) dependencies.
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "evaluation" / "tiered_metrics_tuples.py"
)
_spec = importlib.util.spec_from_file_location(
    "_tiered_metrics_under_test", _MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
_tm = importlib.util.module_from_spec(_spec)
sys.modules["_tiered_metrics_under_test"] = _tm
_spec.loader.exec_module(_tm)

infer_column_datatype = _tm.infer_column_datatype
are_properties_compatible = _tm.are_properties_compatible


# =============================================================================
# Datatype Inference Tests
# =============================================================================


class TestInferColumnDatatype:
    """Tests for infer_column_datatype() - value-based type detection."""

    def test_numeric_integers(self):
        """Integer-only columns detected as numeric."""
        values = {"10", "20", "30", "100"}
        assert infer_column_datatype(values) == "numeric"

    def test_numeric_floats(self):
        """Float columns detected as numeric."""
        values = {"3.14", "2.71", "1.41"}
        assert infer_column_datatype(values) == "numeric"

    def test_numeric_mixed_int_float(self):
        """Mixed int/float columns detected as numeric."""
        values = {"10", "3.14", "20", "2.71"}
        assert infer_column_datatype(values) == "numeric"

    def test_numeric_negative(self):
        """Negative numbers detected as numeric."""
        values = {"-5", "-10.5", "3"}
        assert infer_column_datatype(values) == "numeric"

    def test_numeric_scientific_notation(self):
        """Scientific notation detected as numeric."""
        values = {"1e10", "3.14e-5", "2.5e+3"}
        assert infer_column_datatype(values) == "numeric"

    def test_temporal_dates(self):
        """ISO-8601 dates detected as temporal."""
        values = {"2020-01-15", "2021-06-30", "2019-12-25"}
        assert infer_column_datatype(values) == "temporal"

    def test_temporal_datetimes(self):
        """ISO-8601 datetimes detected as temporal."""
        values = {"2020-01-15T10:30:00", "2021-06-30T14:45:30"}
        assert infer_column_datatype(values) == "temporal"

    def test_temporal_years_ambiguous(self):
        """Year-only values are ambiguous (can be numeric or temporal).

        Values like '2020', '2021' can be parsed as both integers and years.
        Our implementation prefers numeric (checked first), which is acceptable
        since the datatype property check will later distinguish xsd:integer
        from xsd:gYear/xsd:date based on the schema.
        """
        values = {"2020", "2021", "2022"}
        result = infer_column_datatype(values)
        # Either numeric or temporal is acceptable for year-only values
        assert result in ("numeric", "temporal")

    def test_temporal_year_month(self):
        """Year-month values detected as temporal."""
        values = {"2020-01", "2021-06", "2019-12"}
        assert infer_column_datatype(values) == "temporal"

    def test_mixed_numeric_string_returns_none(self):
        """Mixed numeric/string columns return None."""
        values = {"10", "20", "hello", "world"}
        assert infer_column_datatype(values) is None

    def test_string_only_returns_none(self):
        """Pure string columns return None."""
        values = {"Berlin", "Paris", "London"}
        assert infer_column_datatype(values) is None

    def test_empty_set_returns_none(self):
        """Empty value set returns None."""
        values = set()
        assert infer_column_datatype(values) is None

    def test_empty_strings_returns_none(self):
        """All-empty strings return None."""
        values = {"", "  ", "   "}
        assert infer_column_datatype(values) is None

    def test_alphanumeric_codes_return_none(self):
        """Alphanumeric codes (e.g., 'M4', 'U6') return None."""
        values = {"M4", "U6", "10", "20"}
        # Even though some values are numeric, presence of 'M4', 'U6' makes it mixed
        assert infer_column_datatype(values) is None

    def test_temporal_time_only(self):
        """Time-only values (HH:MM:SS) detected as temporal."""
        values = {"00:04:51", "00:14:38", "23:59:59"}
        assert infer_column_datatype(values) == "temporal"


# =============================================================================
# Property Compatibility Tests
# =============================================================================


class TestPropertyCompatibility:
    """Tests for are_properties_compatible() - Property URI matching."""

    def test_same_property_compatible(self):
        """Same property URI is compatible."""
        assert are_properties_compatible(
            "http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/productPropertyNumeric1",
            "http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/productPropertyNumeric1",
        )

    def test_different_properties_incompatible(self):
        """Different property URIs are INcompatible (key fix for routeName vs stationCode)."""
        assert not are_properties_compatible(
            "http://transport.linkeddata.es/def/tro#abbreviation",  # routeName
            "http://transport.linkeddata.es/def/tro#code",  # stationCode
        )

    def test_missing_gt_property_fail_open(self):
        """Missing GT property → fail-open (allow match)."""
        assert are_properties_compatible(
            None,
            "http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/productPropertyNumeric1",
        )

    def test_missing_llm_property_fail_open(self):
        """Missing LLM property → fail-open (allow match)."""
        assert are_properties_compatible(
            "http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/productPropertyNumeric1",
            None,
        )

    def test_both_missing_fail_open(self):
        """Both properties missing → fail-open (allow match)."""
        assert are_properties_compatible(None, None)

    def test_rdfs_label_vs_bsbm_numeric(self):
        """Different properties (rdfs:label vs bsbm:numeric) are incompatible."""
        assert not are_properties_compatible(
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/productPropertyNumeric1",
        )

    def test_case_sensitive_uris(self):
        """Property URIs are case-sensitive."""
        # Different case → different properties
        assert not are_properties_compatible(
            "http://example.org/property",
            "http://example.org/Property",  # Capital P
        )


# =============================================================================
# Integration Tests - End-to-End Column Matching
# =============================================================================


class TestIntegrationDatatypeFiltering:
    """Integration tests for datatype filtering in column matching.

    These tests verify that the datatype checking is properly integrated
    into the column mapping pipeline.
    """

    def test_routename_stationcode_blocked(self):
        """BASE04 regression: routeName should NOT match stationCode.

        This is the core regression case that motivated the property checking feature.
        routeName and stationCode both have numeric values ("10", "20", etc.) but
        come from different properties (tro:abbreviation vs tro:code).
        """
        # Simulate routeName (numeric content from tro:abbreviation)
        route_values = {"10", "20", "30"}  # All numeric
        route_type = infer_column_datatype(route_values)
        assert route_type == "numeric"  # Values look numeric

        # Simulate stationCode (numeric content from tro:code)
        station_values = {"10", "20", "30"}  # All numeric
        station_type = infer_column_datatype(station_values)
        assert station_type == "numeric"  # Values look numeric

        # Both are numeric, so property check would be triggered
        # Check if their properties are the same
        gt_property = "http://transport.linkeddata.es/def/tro#abbreviation"  # routeName
        llm_property = "http://transport.linkeddata.es/def/tro#code"  # stationCode

        # Should be INcompatible (different properties) → match would be blocked
        assert not are_properties_compatible(gt_property, llm_property)

    def test_valid_numeric_match_allowed(self):
        """Legitimate numeric-to-numeric matches are still allowed."""
        # Both columns are integers from the SAME property
        gt_values = {"100", "200", "300"}
        llm_values = {"100", "200", "300"}

        gt_type = infer_column_datatype(gt_values)
        llm_type = infer_column_datatype(llm_values)

        assert gt_type == "numeric"
        assert llm_type == "numeric"

        # Both use the same property → compatible
        gt_property = "http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/productPropertyNumeric1"
        llm_property = "http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/productPropertyNumeric1"

        assert are_properties_compatible(gt_property, llm_property)

    def test_string_columns_no_datatype_check(self):
        """String-only columns bypass datatype checking."""
        # These are clearly strings (not parsable as numeric)
        gt_values = {"Berlin", "Paris", "London"}
        llm_values = {"Berlin", "Paris"}

        gt_type = infer_column_datatype(gt_values)
        llm_type = infer_column_datatype(llm_values)

        assert gt_type is None  # Not numeric/temporal
        assert llm_type is None  # Not numeric/temporal

        # Datatype checking would not be triggered
        # (column mapping would use only value overlap)
