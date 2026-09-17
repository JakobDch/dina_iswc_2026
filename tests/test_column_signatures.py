"""Tests for column signature extraction and matching.

The tests cover the four representative BASE queries from the experimental
corpus plus the crucial regression case of BASE04 where a LLM-emitted
``?stationCode`` column must NOT be mapped to GT's ``?routeName`` despite
coincidental numeric value overlap.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the module directly to avoid triggering src/evaluation/__init__.py, which
# imports agent modules with heavy (and currently broken) dependencies.
_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "evaluation" / "column_signatures.py"
)
_spec = importlib.util.spec_from_file_location(
    "_column_signatures_under_test", _MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
_cs = importlib.util.module_from_spec(_spec)
sys.modules["_column_signatures_under_test"] = _cs
_spec.loader.exec_module(_cs)

ColumnSignature = _cs.ColumnSignature
SignatureMatch = _cs.SignatureMatch
extract_column_signatures = _cs.extract_column_signatures
match_columns = _cs.match_columns
signature_similarity = _cs.signature_similarity


# =============================================================================
# Reference queries (kept inline for test independence from the live corpus)
# =============================================================================


GT_BASE01 = """
PREFIX eduo: <http://example.org/ontology/education#>
SELECT DISTINCT ?student ?studentName ?course ?courseName ?prof ?profName ?dept ?deptName WHERE {
    ?student a eduo:Learner .
    ?student eduo:enrolledIn ?course .
    ?student eduo:belongsTo ?dept .
    ?prof eduo:instructorOf ?course .
    ?prof eduo:employedAt ?dept .
    OPTIONAL { ?student eduo:label ?studentName }
    OPTIONAL { ?course eduo:label ?courseName }
    OPTIONAL { ?prof eduo:label ?profName }
    OPTIONAL { ?dept eduo:label ?deptName }
}
LIMIT 5000
"""

GT_BASE04 = """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT DISTINCT ?route ?routeName ?routeLongName ?stop ?stopName ?arrivalTime WHERE {
    ?route a tro:Line .
    ?trip tro:line ?route .
    ?stopTime a tro:StopEvent .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:reachTime ?arrivalTime .
    OPTIONAL { ?route tro:abbreviation ?routeName }
    OPTIONAL { ?route tro:fullTitle ?routeLongName }
    OPTIONAL { ?stop foaf:name ?stopName }
}
LIMIT 1000
"""

LLM_BASE04 = """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?line ?lineTitle ?station ?stationCode ?stopSequence WHERE {
    ?stopEvent a tro:StopEvent .
    ?stopEvent tro:journey ?trip .
    ?stopEvent tro:station ?station .
    ?stopEvent tro:stopSequence ?stopSequence .
    ?trip tro:line ?line .
    ?line tro:fullTitle ?lineTitle .
    ?station tro:code ?stationCode .
}
LIMIT 1000
"""

GT_BASE06 = """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?route ?routeName ?routeLongName
       (SUM(?depsPerTrip) AS ?totalDepartures)
       (AVG(?depsPerTrip) AS ?avgDeparturesPerTrip)
WHERE {
    {
        SELECT ?route ?trip (COUNT(?stopTime) AS ?depsPerTrip) WHERE {
            ?route a tro:Line .
            ?trip tro:line ?route .
            ?stopTime a tro:StopEvent .
            ?stopTime tro:journey ?trip .
            ?stopTime tro:leaveTime ?depTime .
        }
        GROUP BY ?route ?trip
    }
    OPTIONAL { ?route tro:abbreviation ?routeName }
    OPTIONAL { ?route tro:fullTitle ?routeLongName }
}
GROUP BY ?route ?routeName ?routeLongName
ORDER BY DESC(?totalDepartures)
"""

LLM_BASE06 = """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?line ?lineName ?totalStopEvents (AVG(?stopEventsPerJourney) AS ?avgStopEventsPerJourney)
WHERE {
    {
        SELECT ?line (COUNT(?stopEvent) AS ?totalStopEvents) WHERE {
            ?stopEvent a tro:StopEvent .
            ?stopEvent tro:journey ?trip .
            ?trip tro:line ?line .
        }
        GROUP BY ?line
    }
    {
        SELECT ?line (COUNT(?stopEvent) AS ?stopEventsPerJourney) WHERE {
            ?stopEvent a tro:StopEvent .
            ?stopEvent tro:journey ?trip .
            ?trip tro:line ?line .
        }
        GROUP BY ?line ?trip
    }
    ?line tro:fullTitle ?lineName .
}
GROUP BY ?line ?lineName ?totalStopEvents
"""

GT_BASE08 = """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stop ?stopName (COUNT(?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    OPTIONAL { ?stop foaf:name ?stopName }
}
GROUP BY ?stop ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
"""


# =============================================================================
# Signature extraction
# =============================================================================


class TestBase04Extraction:
    """BASE04 — BGP-only, no aggregations."""

    def test_all_selected_vars_have_signatures(self):
        sigs = extract_column_signatures(GT_BASE04)
        expected = {"route", "routeName", "routeLongName", "stop", "stopName", "arrivalTime"}
        assert set(sigs.keys()) == expected

    def test_route_is_uri_with_direct_type(self):
        sigs = extract_column_signatures(GT_BASE04)
        route = sigs["route"]
        assert route.is_uri is True
        assert "http://example.org/ontology/transport#Line" in route.direct_types
        assert "http://example.org/ontology/transport#line" in route.incoming_predicates
        assert "http://example.org/ontology/transport#abbreviation" in route.outgoing_predicates
        assert "http://example.org/ontology/transport#fullTitle" in route.outgoing_predicates

    def test_routeName_is_literal_from_abbreviation(self):
        sigs = extract_column_signatures(GT_BASE04)
        rn = sigs["routeName"]
        assert rn.is_uri is False
        assert rn.incoming_predicates == {
            "http://example.org/ontology/transport#abbreviation"
        }
        assert not rn.outgoing_predicates
        assert not rn.direct_types

    def test_routeLongName_is_literal_from_fullTitle(self):
        sigs = extract_column_signatures(GT_BASE04)
        rln = sigs["routeLongName"]
        assert rln.is_uri is False
        assert rln.incoming_predicates == {
            "http://example.org/ontology/transport#fullTitle"
        }

    def test_stop_is_uri_incoming_station(self):
        sigs = extract_column_signatures(GT_BASE04)
        stop = sigs["stop"]
        # `stop` is subject of foaf:name (OPTIONAL) and object of tro:station
        assert stop.is_uri is True
        assert "http://example.org/ontology/transport#station" in stop.incoming_predicates
        assert "http://xmlns.com/foaf/0.1/name" in stop.outgoing_predicates

    def test_arrivalTime_incoming_reachTime(self):
        sigs = extract_column_signatures(GT_BASE04)
        at = sigs["arrivalTime"]
        assert at.is_uri is False
        assert at.incoming_predicates == {
            "http://example.org/ontology/transport#reachTime"
        }


class TestBase04LLMExtraction:
    """The agent-emitted BASE04 query used in the failing evaluation."""

    def test_stationCode_literal_from_code(self):
        sigs = extract_column_signatures(LLM_BASE04)
        sc = sigs["stationCode"]
        assert sc.is_uri is False
        assert sc.incoming_predicates == {
            "http://example.org/ontology/transport#code"
        }

    def test_lineTitle_literal_from_fullTitle(self):
        sigs = extract_column_signatures(LLM_BASE04)
        lt = sigs["lineTitle"]
        assert lt.is_uri is False
        assert lt.incoming_predicates == {
            "http://example.org/ontology/transport#fullTitle"
        }

    def test_line_and_station_are_uris(self):
        sigs = extract_column_signatures(LLM_BASE04)
        assert sigs["line"].is_uri is True
        assert sigs["station"].is_uri is True


class TestBase06Extraction:
    """BASE06 — aggregations at outer level, subquery with COUNT."""

    def test_totalDepartures_is_aggregated_sum(self):
        sigs = extract_column_signatures(GT_BASE06)
        td = sigs["totalDepartures"]
        assert td.is_aggregated is True
        assert td.aggregation_kind == "SUM"

    def test_avgDeparturesPerTrip_is_aggregated_avg(self):
        sigs = extract_column_signatures(GT_BASE06)
        av = sigs["avgDeparturesPerTrip"]
        assert av.is_aggregated is True
        assert av.aggregation_kind == "AVG"

    def test_non_aggregated_vars_still_have_bgp_signatures(self):
        sigs = extract_column_signatures(GT_BASE06)
        assert sigs["route"].is_aggregated is False
        # Through its appearance in the subquery's BGP, `route` should still
        # appear as URI subject (subqueries are part of the algebra tree).
        # Note: route's role may vary with rdflib's subquery representation;
        # we only assert that incoming tro:line is seen.
        assert "http://example.org/ontology/transport#line" in sigs["route"].incoming_predicates


class TestBase08Extraction:
    """BASE08 — outer COUNT with GROUP BY."""

    def test_tripCount_is_aggregated_count(self):
        sigs = extract_column_signatures(GT_BASE08)
        tc = sigs["tripCount"]
        assert tc.is_aggregated is True
        assert tc.aggregation_kind == "COUNT"

    def test_stop_has_station_incoming(self):
        sigs = extract_column_signatures(GT_BASE08)
        assert "http://example.org/ontology/transport#station" in sigs["stop"].incoming_predicates


class TestBase01Extraction:
    """BASE01 — 8 SELECT vars, all via OPTIONAL labels + BGP joins."""

    def test_label_columns_incoming_label(self):
        sigs = extract_column_signatures(GT_BASE01)
        for label_var in ["studentName", "courseName", "profName", "deptName"]:
            assert sigs[label_var].is_uri is False
            assert sigs[label_var].incoming_predicates == {
                "http://example.org/ontology/education#label"
            }

    def test_uri_columns_have_outgoing_label(self):
        sigs = extract_column_signatures(GT_BASE01)
        for uri_var in ["student", "course", "prof", "dept"]:
            assert sigs[uri_var].is_uri is True
            assert "http://example.org/ontology/education#label" in sigs[uri_var].outgoing_predicates


# =============================================================================
# Similarity scoring
# =============================================================================


class TestSimilarity:
    """signature_similarity: unit tests on synthetic signatures."""

    def test_identical_literals_same_predicate_full_score(self):
        a = ColumnSignature("x", is_uri=False, incoming_predicates={"p1"})
        b = ColumnSignature("y", is_uri=False, incoming_predicates={"p1"})
        assert signature_similarity(a, b) == 1.0

    def test_different_predicates_zero(self):
        a = ColumnSignature("x", is_uri=False, incoming_predicates={"p1"})
        b = ColumnSignature("y", is_uri=False, incoming_predicates={"p2"})
        assert signature_similarity(a, b) == 0.0

    def test_is_uri_mismatch_no_predicates_still_zero(self):
        """Empty signatures score 0.0 regardless of is_uri — no predicates to match."""
        a = ColumnSignature("x", is_uri=True)
        b = ColumnSignature("y", is_uri=False)
        assert signature_similarity(a, b) == 0.0

    def test_is_uri_mismatch_shared_incoming_still_matches(self):
        """is_uri difference must NOT block a match when the producing property
        is the same. This is the key fix for TYPO04/SYN04 regressions where
        the LLM omits outgoing triples from a URI variable."""
        gt = ColumnSignature(
            "route", is_uri=True,
            incoming_predicates={"tro:line"},
            outgoing_predicates={"tro:abbreviation"},
        )
        llm = ColumnSignature(
            "line", is_uri=False,  # only object, never subject
            incoming_predicates={"tro:line"},
        )
        assert signature_similarity(gt, llm) == 1.0

    def test_direction_aware_no_cross_match(self):
        """A URI with outgoing={label} must NOT match a literal with
        incoming={label}. Same property URI but different structural role:
        'dept produces labels' vs 'courseLabel is a label'."""
        dept = ColumnSignature(
            "dept", is_uri=True,
            outgoing_predicates={"eduo:label"},
            incoming_predicates={"eduo:belongsTo"},
        )
        course_label = ColumnSignature(
            "courseLabel", is_uri=False,
            incoming_predicates={"eduo:label"},
        )
        assert signature_similarity(dept, course_label) == 0.0

    def test_subject_context_disambiguates_shared_label(self):
        """When two GT columns share the same producing property (eno:designation),
        the one whose subject matches the LLM subject must score higher.
        This fixes BASE13: gt:name(field) vs gt:operatorName(operator) both
        matching llm:operatorName(operator) at 1.0 → greedy picks wrong one."""
        # GT: ?field eno:designation ?name — subject is Deposit with many predicates
        gt_name = ColumnSignature(
            "name", is_uri=False,
            incoming_predicates={"eno:designation"},
            subject_context={"out:eno:activeDepositOperator", "out:eno:condition",
                             "out:eno:designation", "type:eno:Deposit"},
        )
        # GT: ?operator eno:designation ?operatorName — subject is Company
        gt_opname = ColumnSignature(
            "operatorName", is_uri=False,
            incoming_predicates={"eno:designation"},
            subject_context={"out:eno:designation"},
        )
        # LLM: ?operator eno:designation ?operatorName — subject is Company
        llm_opname = ColumnSignature(
            "operatorName", is_uri=False,
            incoming_predicates={"eno:designation"},
            subject_context={"out:eno:designation", "type:eno:Company"},
        )
        # operatorName↔operatorName must score HIGHER than name↔operatorName
        score_correct = signature_similarity(gt_opname, llm_opname)
        score_wrong = signature_similarity(gt_name, llm_opname)
        assert score_correct > score_wrong

    def test_subject_context_on_real_base13(self):
        """Extract real signatures from a BASE13-style query and verify that
        subject_context is populated for literal columns."""
        sparql = """
        PREFIX eno: <http://example.org/ontology/energy#>
        SELECT ?field ?name ?operator ?operatorName WHERE {
            ?field a eno:Deposit .
            ?field eno:designation ?name .
            ?field eno:activeDepositOperator ?operator .
            ?operator eno:designation ?operatorName .
        }
        """
        sigs = extract_column_signatures(sparql)
        # ?name comes from ?field eno:designation ?name → subject is ?field
        assert "type:http://example.org/ontology/energy#Deposit" in sigs["name"].subject_context
        assert "out:http://example.org/ontology/energy#activeDepositOperator" in sigs["name"].subject_context
        # ?operatorName comes from ?operator eno:designation ?operatorName → subject is ?operator
        assert "in:http://example.org/ontology/energy#activeDepositOperator" in sigs["operatorName"].subject_context
        # They share the same incoming predicate but different subject context
        assert sigs["name"].subject_context != sigs["operatorName"].subject_context

    def test_aggregated_vs_non_aggregated_incompatible(self):
        a = ColumnSignature("x", is_aggregated=True, aggregation_kind="SUM")
        b = ColumnSignature("y", is_aggregated=False, incoming_predicates={"p1"})
        assert signature_similarity(a, b) == 0.0

    def test_sum_vs_count_compatible(self):
        """SUM and COUNT are considered interchangeable for 'total X' queries."""
        a = ColumnSignature(
            "x", is_aggregated=True, aggregation_kind="SUM", aggregation_inner={"p1"}
        )
        b = ColumnSignature(
            "y", is_aggregated=True, aggregation_kind="COUNT", aggregation_inner={"p1"}
        )
        assert signature_similarity(a, b) > 0.0

    def test_sum_vs_avg_incompatible(self):
        a = ColumnSignature(
            "x", is_aggregated=True, aggregation_kind="SUM", aggregation_inner={"p1"}
        )
        b = ColumnSignature(
            "y", is_aggregated=True, aggregation_kind="AVG", aggregation_inner={"p1"}
        )
        assert signature_similarity(a, b) == 0.0

    def test_partial_overlap_gives_partial_score(self):
        a = ColumnSignature(
            "x", is_uri=True,
            outgoing_predicates={"p1", "p2"},
            incoming_predicates={"q1"},
        )
        b = ColumnSignature(
            "y", is_uri=True,
            outgoing_predicates={"p1"},
            incoming_predicates={"q1"},
        )
        score = signature_similarity(a, b)
        assert 0.0 < score <= 1.0


# =============================================================================
# Column matching — the regression case
# =============================================================================


class TestBase04Matching:
    """The regression: routeName must NOT match stationCode."""

    def test_expected_mappings_present(self):
        gt_sigs = extract_column_signatures(GT_BASE04)
        llm_sigs = extract_column_signatures(LLM_BASE04)
        matches = match_columns(gt_sigs, llm_sigs)
        pairs = {(m.gt_var, m.llm_var) for m in matches}

        # These must be mapped
        assert ("route", "line") in pairs
        assert ("routeLongName", "lineTitle") in pairs
        assert ("stop", "station") in pairs

    def test_routeName_not_matched_to_stationCode(self):
        """Core regression: no value-coincidence should produce this pairing."""
        gt_sigs = extract_column_signatures(GT_BASE04)
        llm_sigs = extract_column_signatures(LLM_BASE04)
        matches = match_columns(gt_sigs, llm_sigs)
        pairs = {(m.gt_var, m.llm_var) for m in matches}
        assert ("routeName", "stationCode") not in pairs

    def test_orphan_gt_columns_are_unmatched(self):
        """routeName, stopName, arrivalTime have no LLM counterpart."""
        gt_sigs = extract_column_signatures(GT_BASE04)
        llm_sigs = extract_column_signatures(LLM_BASE04)
        matches = match_columns(gt_sigs, llm_sigs)
        matched_gt = {m.gt_var for m in matches}
        for orphan in ["routeName", "stopName", "arrivalTime"]:
            assert orphan not in matched_gt

    def test_orphan_llm_columns_are_unmatched(self):
        """stationCode, stopSequence have no GT counterpart."""
        gt_sigs = extract_column_signatures(GT_BASE04)
        llm_sigs = extract_column_signatures(LLM_BASE04)
        matches = match_columns(gt_sigs, llm_sigs)
        matched_llm = {m.llm_var for m in matches}
        for orphan in ["stationCode", "stopSequence"]:
            assert orphan not in matched_llm


class TestBase06Matching:
    """BASE06 — aggregation variables should match structurally."""

    def test_uri_and_title_match(self):
        gt_sigs = extract_column_signatures(GT_BASE06)
        llm_sigs = extract_column_signatures(LLM_BASE06)
        matches = match_columns(gt_sigs, llm_sigs)
        pairs = {(m.gt_var, m.llm_var) for m in matches}
        assert ("route", "line") in pairs
        assert ("routeLongName", "lineName") in pairs

    def test_aggregations_match(self):
        gt_sigs = extract_column_signatures(GT_BASE06)
        llm_sigs = extract_column_signatures(LLM_BASE06)
        matches = match_columns(gt_sigs, llm_sigs)
        pairs = {(m.gt_var, m.llm_var) for m in matches}
        # AVG ↔ AVG (both AVG of a COUNT of stopEvents)
        assert ("avgDeparturesPerTrip", "avgStopEventsPerJourney") in pairs
        # SUM/COUNT compatibility: totalDepartures (SUM of COUNT) ↔ totalStopEvents (COUNT)
        assert ("totalDepartures", "totalStopEvents") in pairs

    def test_routeName_unmatched_when_llm_omits_abbreviation(self):
        gt_sigs = extract_column_signatures(GT_BASE06)
        llm_sigs = extract_column_signatures(LLM_BASE06)
        matches = match_columns(gt_sigs, llm_sigs)
        matched_gt = {m.gt_var for m in matches}
        assert "routeName" not in matched_gt


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def test_malformed_query_returns_empty(self):
        assert extract_column_signatures("NOT A SPARQL QUERY") == {}

    def test_empty_select_star_returns_something(self):
        sparql = """
        PREFIX ex: <http://example.org/>
        SELECT * WHERE { ?s ex:p ?o . }
        """
        sigs = extract_column_signatures(sparql)
        assert "s" in sigs
        assert "o" in sigs

    def test_empty_signatures_dont_match(self):
        a = ColumnSignature("x")  # completely empty
        b = ColumnSignature("y")
        # Even though jaccard of empty sets is 0, match_columns must skip them
        result = match_columns({"x": a}, {"y": b})
        assert result == []
