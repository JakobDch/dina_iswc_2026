"""
Experimental Query Corpus for VKGQA Evaluation.

This module implements a controlled experimental design for evaluating
LLM-based SPARQL generation with systematic ambiguity testing.

EXPERIMENTAL DESIGN:
====================

SET BASE (17 queries): Baseline queries using ontology/mapping terminology
  - 11 Schema queries (no specific entity names)
  - 6 Instance queries (named entities, exact formats)
  - EDU/TRN/NRG queries use readable forms of mapping terms (e.g., "learner", "division", "deposit")

SET SYN (17 queries): Natural language synonym variants (same GT as BASE)
  - Tests whether natural language phrasing can be mapped to ontology terms
  - EDU/TRN/NRG: uses everyday language (e.g., "student", "department", "field")

SET TYPO (17 queries): Typo variants of BASE (same GT as BASE)
  - Tests fuzzy matching and error tolerance on mapping terminology

SET UNDER (5 queries): Underspecified queries
  - Vague terms requiring interpretation (e.g., "top", "recent", "large")

SET CROSS (5 queries): Cross-dataset queries (NO other ambiguities)
  - Tests multi-endpoint coordination

TOTAL: 61 queries
  - 51 paired (BASE + SYN + TYPO) for within-subject comparison
  - 10 standalone for specific capability testing

NOTE: Multi-step queries (SET_MULTI) moved to experimental_corpus_multi_backup.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from data.queries.use_case_queries_tiered_tuples import (
    AdaptiveGroundTruth,
    ColumnDef,
    RelevanceLevel,
)


class AmbiguityLayer(str, Enum):
    """Ambiguity layer for experimental queries."""
    BASELINE = "baseline"
    SYNONYM = "synonym"
    TYPO = "typo"
    SEMANTIC = "semantic"
    UNDERSPECIFIED = "underspecified"
    CROSS_DATASET = "cross_dataset"
    MULTI_STEP = "multi_step"


@dataclass
class ExperimentalQuery:
    """Wrapper for experimental queries with metadata.

    Links variant queries (SYN, TYPO) back to their baseline
    for paired comparison analysis.
    """
    ground_truth: AdaptiveGroundTruth
    layer: AmbiguityLayer
    baseline_id: str | None = None  # For SYN/TYPO: links to BASE query
    variant_description: str = ""   # What changed from baseline


# =============================================================================
# SET BASE: Baseline Queries (20 total)
# =============================================================================

# -----------------------------------------------------------------------------
# Schema Queries (12) - No specific entity names
# -----------------------------------------------------------------------------

# EDU complex query - students taking courses taught by professors in their department
BASE01 = AdaptiveGroundTruth(
    query_id="BASE01",
    query_set="BASE",
    description="Students taking courses from own department",
    dataset="EDU",
    datasets=["EDU"],
    query="Show me 5000 examples of learners enrolled in courses instructed by educators employed at their own division.",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
SELECT DISTINCT ?student ?studentName ?course ?courseName ?prof ?profName ?dept ?deptName WHERE {
    ?student a eduo:Learner .
    ?course a eduo:Course .
    ?prof a eduo:Educator .
    ?dept a eduo:Division .
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
    """.strip(), """
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
    """.strip()],
    columns=[
        ColumnDef("student", RelevanceLevel.PREFERRED, "Student URI", semantic_concept="student"),
        ColumnDef("studentName", RelevanceLevel.PREFERRED, "Student name", semantic_concept="student"),
        ColumnDef("course", RelevanceLevel.PREFERRED, "Course URI", semantic_concept="course"),
        ColumnDef("courseName", RelevanceLevel.PREFERRED, "Course name", semantic_concept="course"),
        ColumnDef("prof", RelevanceLevel.ACCEPTABLE, "Professor URI", semantic_concept="professor"),
        ColumnDef("profName", RelevanceLevel.ACCEPTABLE, "Professor name", semantic_concept="professor"),
        ColumnDef("dept", RelevanceLevel.ACCEPTABLE, "Department URI", semantic_concept="department"),
        ColumnDef("deptName", RelevanceLevel.ACCEPTABLE, "Department name", semantic_concept="department"),
    ],
    notes="Complex query: 5 triple patterns with JOIN condition. Two variants: with/without implicit type constraints.",
)

# LCA dataset - activities with their flows
BASE03 = AdaptiveGroundTruth(
    query_id="BASE03",
    query_set="BASE",
    description="Activities with flows",
    dataset="LCA",
    datasets=["LCA"],
    query="List activities with their input and output flows.",
    sparql_queries=["""
PREFIX oriont: <https://orienting.eu/oriont#>
PREFIX oriont-ext: <https://orienting.eu/oriont-ext#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?activity ?activityLabel ?flow ?flowName ?value ?unit WHERE {
    ?activity a oriont:Activity .
    ?activity rdfs:label ?activityLabel .
    { ?flow oriont:isInputOf ?activity } UNION { ?flow oriont:isOutputOf ?activity }
    OPTIONAL { ?flow oriont-ext:hasFlowName ?flowName }
    OPTIONAL { ?flow oriont-ext:hasMeasureValue ?value }
    OPTIONAL { ?flow oriont-ext:hasMeasureUnit ?unit }
}
    """.strip()],
    columns=[
        ColumnDef("activity", RelevanceLevel.PREFERRED, "Activity URI", semantic_concept="activity"),
        ColumnDef("activityLabel", RelevanceLevel.PREFERRED, "Activity name", semantic_concept="activity"),
        ColumnDef("flow", RelevanceLevel.PREFERRED, "Flow URI", semantic_concept="flow"),
        ColumnDef("flowName", RelevanceLevel.PREFERRED, "Flow name", semantic_concept="flow"),
        ColumnDef("value", RelevanceLevel.ACCEPTABLE, "Measure value", semantic_concept="value"),
        ColumnDef("unit", RelevanceLevel.ACCEPTABLE, "Unit", semantic_concept="value"),
    ],
    notes="Baseline: BGP with UNION (isInputOf/isOutputOf) and OPTIONAL. LCA dataset.",
)

# From B03 in use_case_queries_tiered_tuples.py (verified/tested)
BASE04 = AdaptiveGroundTruth(
    query_id="BASE04",
    query_set="BASE",
    description="Routes and stops",
    dataset="TRN",
    datasets=["TRN"],
    query="Show me 1000 examples of which lines have stop events at which stations.",
    sparql_queries=[
        # Variant 1: SELECT DISTINCT, with implicit types
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT DISTINCT ?route ?routeName ?routeLongName ?stop ?stopName ?arrivalTime WHERE {
    ?route a tro:Line .
    ?trip a tro:Trip .
    ?stop a tro:Stop .
    ?stopTime a tro:StopEvent .
    ?trip tro:line ?route .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:reachTime ?arrivalTime .
    OPTIONAL { ?route tro:abbreviation ?routeName }
    OPTIONAL { ?route tro:fullTitle ?routeLongName }
    OPTIONAL { ?stop foaf:name ?stopName }
}
LIMIT 1000
        """.strip(),
        # Variant 2: without DISTINCT, with implicit types
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?route ?routeName ?routeLongName ?stop ?stopName ?arrivalTime WHERE {
    ?route a tro:Line .
    ?trip a tro:Trip .
    ?stop a tro:Stop .
    ?stopTime a tro:StopEvent .
    ?trip tro:line ?route .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:reachTime ?arrivalTime .
    OPTIONAL { ?route tro:abbreviation ?routeName }
    OPTIONAL { ?route tro:fullTitle ?routeLongName }
    OPTIONAL { ?stop foaf:name ?stopName }
}
LIMIT 1000
        """.strip(),
        # Variant 3: DISTINCT without implicit types
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT DISTINCT ?route ?routeName ?routeLongName ?stop ?stopName ?arrivalTime WHERE {
    ?route a tro:Line .
    ?stopTime a tro:StopEvent .
    ?trip tro:line ?route .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:reachTime ?arrivalTime .
    OPTIONAL { ?route tro:abbreviation ?routeName }
    OPTIONAL { ?route tro:fullTitle ?routeLongName }
    OPTIONAL { ?stop foaf:name ?stopName }
}
LIMIT 1000
        """.strip(),
        # Variant 4: without DISTINCT, without implicit types
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?route ?routeName ?routeLongName ?stop ?stopName ?arrivalTime WHERE {
    ?route a tro:Line .
    ?stopTime a tro:StopEvent .
    ?trip tro:line ?route .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:reachTime ?arrivalTime .
    OPTIONAL { ?route tro:abbreviation ?routeName }
    OPTIONAL { ?route tro:fullTitle ?routeLongName }
    OPTIONAL { ?stop foaf:name ?stopName }
}
LIMIT 1000
        """.strip(),
    ],
    columns=[
        ColumnDef("route", RelevanceLevel.PREFERRED, "Route URI", semantic_concept="route"),
        ColumnDef("routeName", RelevanceLevel.PREFERRED, "Route short name", semantic_concept="route"),
        ColumnDef("routeLongName", RelevanceLevel.PREFERRED, "Route long name", semantic_concept="route"),
        ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
        ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
        ColumnDef("arrivalTime", RelevanceLevel.ACCEPTABLE, "Arrival time", semantic_concept="arrivalTime"),
    ],
    notes="Baseline: Multi-hop BGP. From B03. Two variants: DISTINCT vs non-DISTINCT (duplicates from multiple trips per route-stop pair consume LIMIT slots).",
)

# EDU complex query - students per department with course info
BASE05 = AdaptiveGroundTruth(
    query_id="BASE05",
    query_set="BASE",
    description="Students per department taking courses",
    dataset="EDU",
    datasets=["EDU"],
    query="How many learners in each division are enrolled in courses, and what courses are they enrolled in?",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
SELECT ?dept ?deptName ?studentCount ?courseCount ?courses ?courseNames WHERE {
    {
        SELECT ?dept (COUNT(DISTINCT ?student) AS ?studentCount) (COUNT(DISTINCT ?course) AS ?courseCount)
               (GROUP_CONCAT(DISTINCT STR(?course); SEPARATOR=", ") AS ?courses)
               (GROUP_CONCAT(DISTINCT ?courseName; SEPARATOR=", ") AS ?courseNames)
        WHERE {
            ?dept a eduo:Division .
            ?student a eduo:Learner .
            ?course a eduo:Course .
            ?student eduo:belongsTo ?dept .
            ?student eduo:enrolledIn ?course .
            OPTIONAL { ?course eduo:label ?courseName }
        }
        GROUP BY ?dept
    }
    OPTIONAL { ?dept eduo:label ?deptName }
}
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
SELECT ?dept ?deptName ?studentCount ?courseCount ?courses ?courseNames WHERE {
    {
        SELECT ?dept (COUNT(DISTINCT ?student) AS ?studentCount) (COUNT(DISTINCT ?course) AS ?courseCount)
               (GROUP_CONCAT(DISTINCT STR(?course); SEPARATOR=", ") AS ?courses)
               (GROUP_CONCAT(DISTINCT ?courseName; SEPARATOR=", ") AS ?courseNames)
        WHERE {
            ?dept a eduo:Division .
            ?student a eduo:Learner .
            ?student eduo:belongsTo ?dept .
            ?student eduo:enrolledIn ?course .
            OPTIONAL { ?course eduo:label ?courseName }
        }
        GROUP BY ?dept
    }
    OPTIONAL { ?dept eduo:label ?deptName }
}
    """.strip()],
    columns=[
        ColumnDef("dept", RelevanceLevel.PREFERRED, "Department URI", semantic_concept="department"),
        ColumnDef("deptName", RelevanceLevel.PREFERRED, "Department name", semantic_concept="department"),
        ColumnDef("studentCount", RelevanceLevel.PREFERRED, "Student count", is_measurement=True, semantic_concept="count"),
        ColumnDef("courseCount", RelevanceLevel.PREFERRED, "Course count", is_measurement=True, semantic_concept="count"),
        ColumnDef("courses", RelevanceLevel.PREFERRED, "Course URIs", semantic_concept="course"),
        ColumnDef("courseNames", RelevanceLevel.PREFERRED, "Course names", semantic_concept="course"),
    ],
    notes="Complex query: Subquery with COUNT + GROUP_CONCAT, OPTIONALs for names. Answers both 'how many' and 'what courses'.",
)

# TRN aggregation query - tests SUM and AVG with subquery pattern
BASE06 = AdaptiveGroundTruth(
    query_id="BASE06",
    query_set="BASE",
    description="Total and average departures per route",
    dataset="TRN",
    datasets=["TRN"],
    query="What is the total number of stop events and average stop events per journey for each line?",
    sparql_queries=[
        # Variant 1: Count only stop events WITH departure time (leaveTime)
        """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?route ?routeName ?routeLongName
       (SUM(?depsPerTrip) AS ?totalDepartures)
       (AVG(?depsPerTrip) AS ?avgDeparturesPerTrip)
WHERE {
    {
        SELECT ?route ?trip (COUNT(?stopTime) AS ?depsPerTrip) WHERE {
            ?route a tro:Line .
            ?trip a tro:Trip .
            ?stopTime a tro:StopEvent .
            ?trip tro:line ?route .
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
        """.strip(),
        # Variant 2: Count ALL stop events (no leaveTime filter — includes terminal stops)
        """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?route ?routeName ?routeLongName
       (SUM(?stopsPerTrip) AS ?totalStopEvents)
       (AVG(?stopsPerTrip) AS ?avgStopEventsPerTrip)
WHERE {
    {
        SELECT ?route ?trip (COUNT(?stopTime) AS ?stopsPerTrip) WHERE {
            ?route a tro:Line .
            ?trip a tro:Trip .
            ?stopTime a tro:StopEvent .
            ?trip tro:line ?route .
            ?stopTime tro:journey ?trip .
        }
        GROUP BY ?route ?trip
    }
    OPTIONAL { ?route tro:abbreviation ?routeName }
    OPTIONAL { ?route tro:fullTitle ?routeLongName }
}
GROUP BY ?route ?routeName ?routeLongName
ORDER BY DESC(?totalStopEvents)
        """.strip(),
        # Variant 3: V1 without implicit ?trip type
        """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?route ?routeName ?routeLongName
       (SUM(?depsPerTrip) AS ?totalDepartures)
       (AVG(?depsPerTrip) AS ?avgDeparturesPerTrip)
WHERE {
    {
        SELECT ?route ?trip (COUNT(?stopTime) AS ?depsPerTrip) WHERE {
            ?route a tro:Line .
            ?stopTime a tro:StopEvent .
            ?trip tro:line ?route .
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
        """.strip(),
        # Variant 4: V2 without implicit ?trip type
        """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?route ?routeName ?routeLongName
       (SUM(?stopsPerTrip) AS ?totalStopEvents)
       (AVG(?stopsPerTrip) AS ?avgStopEventsPerTrip)
WHERE {
    {
        SELECT ?route ?trip (COUNT(?stopTime) AS ?stopsPerTrip) WHERE {
            ?route a tro:Line .
            ?stopTime a tro:StopEvent .
            ?trip tro:line ?route .
            ?stopTime tro:journey ?trip .
        }
        GROUP BY ?route ?trip
    }
    OPTIONAL { ?route tro:abbreviation ?routeName }
    OPTIONAL { ?route tro:fullTitle ?routeLongName }
}
GROUP BY ?route ?routeName ?routeLongName
ORDER BY DESC(?totalStopEvents)
        """.strip(),
    ],
    columns=[
        ColumnDef("route", RelevanceLevel.PREFERRED, "Route URI", semantic_concept="route"),
        ColumnDef("routeName", RelevanceLevel.PREFERRED, "Route short name", semantic_concept="route"),
        ColumnDef("routeLongName", RelevanceLevel.PREFERRED, "Route long name", semantic_concept="route"),
        ColumnDef("totalDepartures", RelevanceLevel.PREFERRED, "Total departures", is_measurement=True, semantic_concept="departures"),
        ColumnDef("avgDeparturesPerTrip", RelevanceLevel.PREFERRED, "Average departures per trip", is_measurement=True, semantic_concept="departures"),
    ],
    notes="Baseline: Subquery with GROUP BY, SUM and AVG. Two variants: with leaveTime filter (departures only) vs. all stop events (including terminal stops).",
)

# From A07 in use_case_queries_tiered_tuples.py (verified/tested) - adjusted LIMIT to 10
BASE08 = AdaptiveGroundTruth(
    query_id="BASE08",
    query_set="BASE",
    description="Top 10 stops by trips",
    dataset="TRN",
    datasets=["TRN"],
    query="Show the top 10 stations with the most journeys.",
    sparql_queries=[
        # Variant 1: COUNT without DISTINCT, GROUP BY URI+name
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stop ?stopName (COUNT(?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stop a tro:Stop .
    ?trip a tro:Trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    OPTIONAL { ?stop foaf:name ?stopName }
}
GROUP BY ?stop ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
        # Variant 2: COUNT DISTINCT, GROUP BY URI+name
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stop ?stopName (COUNT(DISTINCT ?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stop a tro:Stop .
    ?trip a tro:Trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    OPTIONAL { ?stop foaf:name ?stopName }
}
GROUP BY ?stop ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
        # Variant 3: COUNT, GROUP BY name only (merges stops sharing a name across lines)
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stopName (COUNT(?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stop a tro:Stop .
    ?trip a tro:Trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    ?stop foaf:name ?stopName .
}
GROUP BY ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
        # Variant 4: COUNT DISTINCT, GROUP BY name only
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stopName (COUNT(DISTINCT ?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stop a tro:Stop .
    ?trip a tro:Trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    ?stop foaf:name ?stopName .
}
GROUP BY ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
        # Variant 5: V1 untyped (no ?stop/?trip type filter)
        """
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
        """.strip(),
        # Variant 6: V2 untyped
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stop ?stopName (COUNT(DISTINCT ?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    OPTIONAL { ?stop foaf:name ?stopName }
}
GROUP BY ?stop ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
        # Variant 7: typed + COUNT + GROUP BY URI+name + REQUIRED name
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stop ?stopName (COUNT(?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stop a tro:Stop .
    ?trip a tro:Trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    ?stop foaf:name ?stopName .
}
GROUP BY ?stop ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
        # Variant 8: typed + COUNT DISTINCT + GROUP BY URI+name + REQUIRED name
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stop ?stopName (COUNT(DISTINCT ?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stop a tro:Stop .
    ?trip a tro:Trip .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    ?stop foaf:name ?stopName .
}
GROUP BY ?stop ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
        # Variant 9: untyped + COUNT + GROUP BY URI+name + REQUIRED name
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stop ?stopName (COUNT(?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    ?stop foaf:name ?stopName .
}
GROUP BY ?stop ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
        # Variant 10: untyped + COUNT DISTINCT + GROUP BY URI+name + REQUIRED name
        """
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?stop ?stopName (COUNT(DISTINCT ?trip) AS ?tripCount) WHERE {
    ?stopTime a tro:StopEvent .
    ?stopTime tro:station ?stop .
    ?stopTime tro:journey ?trip .
    ?stop foaf:name ?stopName .
}
GROUP BY ?stop ?stopName
ORDER BY DESC(?tripCount)
LIMIT 10
        """.strip(),
    ],
    columns=[
        [  # Variant 1: GROUP BY URI+name
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 2
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 3: GROUP BY name only (no URI)
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 4
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 5: V1 untyped (GROUP BY URI+name, OPTIONAL name)
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 6: V2 untyped
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 7: typed + COUNT + REQUIRED name
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 8: typed + COUNT DISTINCT + REQUIRED name
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 9: untyped + COUNT + REQUIRED name
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # Variant 10: untyped + COUNT DISTINCT + REQUIRED name
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("stopName", RelevanceLevel.PREFERRED, "Stop name", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.ACCEPTABLE, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
    ],
    notes="Baseline: ORDER BY + LIMIT. 10 variants: COUNT/COUNT DISTINCT × GROUP-by-URI+name/name-only × typed/untyped × OPTIONAL/REQUIRED name.",
)

# EDU complex query - professors teaching graduate courses without advisees
BASE09 = AdaptiveGroundTruth(
    query_id="BASE09",
    query_set="BASE",
    description="Graduate course professors without advisees",
    dataset="EDU",
    datasets=["EDU"],
    query="Which educators who instruct advanced modules have no mentees?",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
SELECT DISTINCT ?prof ?profName ?course ?courseName WHERE {
    ?prof a eduo:Educator .
    ?prof eduo:instructorOf ?course .
    ?course a eduo:AdvancedModule .
    OPTIONAL { ?prof eduo:label ?profName }
    OPTIONAL { ?course eduo:label ?courseName }
    MINUS { ?student eduo:mentor ?prof . }
}
    """.strip()],
    columns=[
        ColumnDef("prof", RelevanceLevel.PREFERRED, "Professor URI", semantic_concept="professor"),
        ColumnDef("profName", RelevanceLevel.PREFERRED, "Professor name", semantic_concept="professor"),
        ColumnDef("course", RelevanceLevel.ACCEPTABLE, "Course URI", semantic_concept="graduateCourse"),
        ColumnDef("courseName", RelevanceLevel.ACCEPTABLE, "Course name", semantic_concept="graduateCourse"),
    ],
    notes="Complex query: 3 triple patterns + MINUS negation.",
)

# MIN/MAX aggregation query - replaces redundant MINUS pattern
BASE10 = AdaptiveGroundTruth(
    query_id="BASE10",
    query_set="BASE",
    description="Cheapest and most expensive products",
    dataset="BSBM",
    datasets=["BSBM"],
    query="What are the cheapest and most expensive products?",
    sparql_queries=["""
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?product ?label ?price ?priceCategory WHERE {
    {
        SELECT ?product ?label ?price ("cheapest" AS ?priceCategory) WHERE {
            ?product a bsbm:Product .
            ?offer a bsbm:Offer .
            ?offer bsbm:product ?product .
            ?offer bsbm:price ?price .
            OPTIONAL { ?product rdfs:label ?label }
        }
        ORDER BY ASC(?price)
        LIMIT 1
    }
    UNION
    {
        SELECT ?product ?label ?price ("most_expensive" AS ?priceCategory) WHERE {
            ?product a bsbm:Product .
            ?offer a bsbm:Offer .
            ?offer bsbm:product ?product .
            ?offer bsbm:price ?price .
            OPTIONAL { ?product rdfs:label ?label }
        }
        ORDER BY DESC(?price)
        LIMIT 1
    }
}
    """.strip(), """
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?product ?label ?price ?priceCategory WHERE {
    {
        SELECT ?product ?label ?price ("cheapest" AS ?priceCategory) WHERE {
            ?product a bsbm:Product .
            ?offer bsbm:product ?product .
            ?offer bsbm:price ?price .
            OPTIONAL { ?product rdfs:label ?label }
        }
        ORDER BY ASC(?price)
        LIMIT 1
    }
    UNION
    {
        SELECT ?product ?label ?price ("most_expensive" AS ?priceCategory) WHERE {
            ?product a bsbm:Product .
            ?offer bsbm:product ?product .
            ?offer bsbm:price ?price .
            OPTIONAL { ?product rdfs:label ?label }
        }
        ORDER BY DESC(?price)
        LIMIT 1
    }
}
    """.strip()],
    columns=[
        ColumnDef("product", RelevanceLevel.PREFERRED, "Product URI", semantic_concept="product"),
        ColumnDef("label", RelevanceLevel.PREFERRED, "Product label", semantic_concept="product"),
        ColumnDef("price", RelevanceLevel.PREFERRED, "Price", is_measurement=True, semantic_concept="price"),
        ColumnDef("priceCategory", RelevanceLevel.ACCEPTABLE, "Price category", semantic_concept="priceCategory", is_bind_label=True),
    ],
    notes="Baseline: UNION with ORDER BY + LIMIT subqueries. Two variants: with/without implicit ?offer type.",
)

# From A13 in use_case_queries_tiered_tuples.py (verified/tested)
BASE11 = AdaptiveGroundTruth(
    query_id="BASE11",
    query_set="BASE",
    description="High-production fields",
    dataset="NRG",
    datasets=["NRG"],
    query="Which deposits have more than 50 million in extracted crude total?",
    sparql_queries=["""
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?field ?fieldName (SUM(?oilProd) AS ?totalOil) WHERE {
    ?prod a eno:AnnualDepositOutput .
    ?field a eno:Deposit .
    ?prod eno:outputForDeposit ?field .
    ?prod eno:extractedCrude ?oilProd .
    OPTIONAL { ?field eno:designation ?fieldName }
}
GROUP BY ?field ?fieldName
HAVING(SUM(?oilProd) > 50)
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?field ?fieldName (SUM(?oilProd) AS ?totalOil) WHERE {
    ?prod a eno:AnnualDepositOutput .
    ?prod eno:outputForDeposit ?field .
    ?prod eno:extractedCrude ?oilProd .
    OPTIONAL { ?field eno:designation ?fieldName }
}
GROUP BY ?field ?fieldName
HAVING(SUM(?oilProd) > 50)
    """.strip()],
    columns=[
        ColumnDef("field", RelevanceLevel.PREFERRED, "Field URI", semantic_concept="field"),
        ColumnDef("fieldName", RelevanceLevel.PREFERRED, "Field name", semantic_concept="field"),
        ColumnDef("totalOil", RelevanceLevel.ACCEPTABLE, "Total oil production", is_measurement=True, semantic_concept="oilProduction"),
    ],
    notes="Baseline: HAVING clause. From A13.",
)

# EDU complex query - all courses with teachers and departments
BASE12 = AdaptiveGroundTruth(
    query_id="BASE12",
    query_set="BASE",
    description="All courses with teachers and departments",
    dataset="EDU",
    datasets=["EDU"],
    query="List all basic modules and advanced modules with their instructors and division.",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
SELECT DISTINCT ?course ?courseName ?prof ?profName ?dept ?deptName WHERE {
    ?course a ?courseType .
    FILTER(?courseType IN (eduo:AdvancedModule, eduo:BasicModule))
    ?prof a eduo:Educator .
    ?dept a eduo:Division .
    ?prof eduo:instructorOf ?course .
    ?prof eduo:employedAt ?dept .
    OPTIONAL { ?course eduo:label ?courseName }
    OPTIONAL { ?prof eduo:label ?profName }
    OPTIONAL { ?dept eduo:label ?deptName }
}
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
SELECT DISTINCT ?course ?courseName ?prof ?profName ?dept ?deptName WHERE {
    ?course a ?courseType .
    FILTER(?courseType IN (eduo:AdvancedModule, eduo:BasicModule))
    ?prof eduo:instructorOf ?course .
    ?prof eduo:employedAt ?dept .
    OPTIONAL { ?course eduo:label ?courseName }
    OPTIONAL { ?prof eduo:label ?profName }
    OPTIONAL { ?dept eduo:label ?deptName }
}
    """.strip()],
    columns=[
        ColumnDef("course", RelevanceLevel.PREFERRED, "Course URI", semantic_concept="course"),
        ColumnDef("courseName", RelevanceLevel.PREFERRED, "Course name", semantic_concept="course"),
        ColumnDef("prof", RelevanceLevel.PREFERRED, "Professor URI", semantic_concept="teacher"),
        ColumnDef("profName", RelevanceLevel.PREFERRED, "Professor name", semantic_concept="teacher"),
        ColumnDef("dept", RelevanceLevel.ACCEPTABLE, "Department URI", semantic_concept="department"),
        ColumnDef("deptName", RelevanceLevel.ACCEPTABLE, "Department name", semantic_concept="department"),
    ],
    notes="Complex query: 4 triple patterns with FILTER. Two variants: with/without implicit ?prof/?dept types.",
)

# -----------------------------------------------------------------------------
# Instance Queries (8) - Named entities, exact formats
# -----------------------------------------------------------------------------

# From C01 in use_case_queries_tiered_tuples.py (verified/tested) - simplified for baseline
BASE13 = AdaptiveGroundTruth(
    query_id="BASE13",
    query_set="BASE",
    description="Troll field data",
    dataset="NRG",
    datasets=["NRG"],
    query="Show the condition and active deposit operator of the TROLL deposit.",
    sparql_queries=["""
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?field ?name ?status ?operator ?operatorName WHERE {
    ?field a eno:Deposit .
    ?field eno:designation ?name .
    FILTER(UCASE(?name) = "TROLL")
    OPTIONAL { ?field eno:condition ?status }
    OPTIONAL {
        ?field eno:activeDepositOperator ?operator .
        ?operator a eno:Company .
        OPTIONAL { ?operator eno:designation ?operatorName }
    }
}
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?field ?name ?status ?operator ?operatorName WHERE {
    ?field a eno:Deposit .
    ?field eno:designation ?name .
    FILTER(UCASE(?name) = "TROLL")
    OPTIONAL { ?field eno:condition ?status }
    OPTIONAL {
        ?field eno:activeDepositOperator ?operator .
        OPTIONAL { ?operator eno:designation ?operatorName }
    }
}
    """.strip()],
    columns=[
        ColumnDef("field", RelevanceLevel.ACCEPTABLE, "Field URI", semantic_concept="TROLL"),
        ColumnDef("name", RelevanceLevel.ACCEPTABLE, "Field name", semantic_concept="TROLL"),
        ColumnDef("status", RelevanceLevel.PREFERRED, "Status", semantic_concept="status"),
        ColumnDef("operator", RelevanceLevel.PREFERRED, "Operator URI", semantic_concept="operator"),
        ColumnDef("operatorName", RelevanceLevel.PREFERRED, "Operator name", semantic_concept="operator"),
    ],
    notes="Baseline: Named entity (exact case). From C01 pattern.",
)

BASE14 = AdaptiveGroundTruth(
    query_id="BASE14",
    query_set="BASE",
    description="Courses at University0",
    dataset="EDU",
    datasets=["EDU"],
    query="Show courses instructed at University0.",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
SELECT DISTINCT ?course ?courseName ?dept ?deptName WHERE {
    ?uni a eduo:Academy .
    ?dept a eduo:Division .
    ?prof a eduo:Educator .
    ?course a eduo:Course .
    ?uni eduo:label "University0" .
    ?dept eduo:partOf ?uni .
    ?prof eduo:employedAt ?dept .
    ?prof eduo:instructorOf ?course .
    OPTIONAL { ?course eduo:label ?courseName }
    OPTIONAL { ?dept eduo:label ?deptName }
}
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
SELECT DISTINCT ?course ?courseName ?dept ?deptName WHERE {
    ?uni eduo:label "University0" .
    ?dept eduo:partOf ?uni .
    ?prof eduo:employedAt ?dept .
    ?prof eduo:instructorOf ?course .
    OPTIONAL { ?course eduo:label ?courseName }
    OPTIONAL { ?dept eduo:label ?deptName }
}
    """.strip()],
    columns=[
        ColumnDef("course", RelevanceLevel.PREFERRED, "Course URI", semantic_concept="course"),
        ColumnDef("courseName", RelevanceLevel.PREFERRED, "Course name", semantic_concept="course"),
        ColumnDef("dept", RelevanceLevel.ACCEPTABLE, "Department URI", semantic_concept="department"),
        ColumnDef("deptName", RelevanceLevel.ACCEPTABLE, "Department name", semantic_concept="department"),
    ],
    notes="Baseline: Named entity (exact match via eduo:label).",
)

# From C09 pattern in use_case_queries_tiered_tuples.py (verified/tested)
BASE16 = AdaptiveGroundTruth(
    query_id="BASE16",
    query_set="BASE",
    description="Fields operated by Statoil",
    dataset="NRG",
    datasets=["NRG"],
    query="Which deposits have Statoil as their active deposit operator?",
    sparql_queries=["""
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?field ?fieldName WHERE {
    ?field a eno:Deposit .
    ?company a eno:Company .
    ?field eno:activeDepositOperator ?company .
    ?company eno:designation ?companyName .
    FILTER(REGEX(?companyName, "STATOIL", "i"))
    OPTIONAL { ?field eno:designation ?fieldName }
}
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?field ?fieldName WHERE {
    ?field a eno:Deposit .
    ?field eno:activeDepositOperator ?company .
    ?company eno:designation ?companyName .
    FILTER(REGEX(?companyName, "STATOIL", "i"))
    OPTIONAL { ?field eno:designation ?fieldName }
}
    """.strip()],
    columns=[
        ColumnDef("field", RelevanceLevel.PREFERRED, "Field URI", semantic_concept="field"),
        ColumnDef("fieldName", RelevanceLevel.PREFERRED, "Field name", semantic_concept="field"),
    ],
    notes="Baseline: Named entity. From C09 pattern.",
)

# From A05 in use_case_queries_tiered_tuples.py (verified/tested)
BASE18 = AdaptiveGroundTruth(
    query_id="BASE18",
    query_set="BASE",
    description="Deep wellbores (>3000m)",
    dataset="NRG",
    datasets=["NRG"],
    query="Find boreholes with a total drill depth greater than 3000 meters.",
    sparql_queries=["""
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?wellbore ?name ?depth WHERE {
    ?wellbore a eno:Borehole .
    ?wellbore eno:totalDrillDepth ?depth .
    FILTER(?depth > 3000)
    OPTIONAL { ?wellbore eno:designation ?name }
}
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?wellbore ?name ?depth WHERE {
    ?wellbore a eno:Borehole .
    ?wellbore eno:finalVerticalDrillDepth ?depth .
    FILTER(?depth > 3000)
    OPTIONAL { ?wellbore eno:designation ?name }
}
    """.strip()],
    columns=[
        ColumnDef("wellbore", RelevanceLevel.PREFERRED, "Wellbore URI", semantic_concept="wellbore"),
        ColumnDef("name", RelevanceLevel.PREFERRED, "Wellbore name", semantic_concept="wellbore"),
        ColumnDef("depth", RelevanceLevel.ACCEPTABLE, "Depth value", semantic_concept="depth"),
    ],
    notes=(
        "Baseline: Numeric FILTER (exact format). From A05. "
        "Two admissible readings: the depth measure is either totalDrillDepth "
        "(along the path) or finalVerticalDrillDepth (vertical). The two "
        "belongsToWell readings added on 2026-04-21 for the former SYN wording "
        "'wells' were removed on 2026-09-14 when SYN18 was reworded to 'drill holes'."
    ),
)

# From A11 in use_case_queries_tiered_tuples.py (verified/tested)
BASE19 = AdaptiveGroundTruth(
    query_id="BASE19",
    query_set="BASE",
    description="Late night trips (00:00-01:00)",
    dataset="TRN",
    datasets=["TRN"],
    query="Show journeys with leave times between 00:00:00 and 01:00:00.",
    sparql_queries=["""
PREFIX tro: <http://example.org/ontology/transport#>
SELECT DISTINCT ?trip ?tripShortName ?headsign ?routeName ?routeLongName ?departureTime WHERE {
    ?stopTime a tro:StopEvent .
    ?trip a tro:Trip .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:leaveTime ?departureTime .
    FILTER(?departureTime >= "00:00:00" && ?departureTime < "01:00:00")
    OPTIONAL { ?trip tro:abbreviation ?tripShortName }
    OPTIONAL { ?trip tro:destination ?headsign }
    OPTIONAL {
        ?trip tro:line ?route .
        ?route a tro:Line .
        ?route tro:abbreviation ?routeName .
        ?route tro:fullTitle ?routeLongName .
    }
}
    """.strip(), """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT DISTINCT ?trip ?tripShortName ?headsign ?routeName ?routeLongName ?departureTime WHERE {
    ?stopTime a tro:StopEvent .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:leaveTime ?departureTime .
    FILTER(?departureTime >= "00:00:00" && ?departureTime < "01:00:00")
    OPTIONAL { ?trip tro:abbreviation ?tripShortName }
    OPTIONAL { ?trip tro:destination ?headsign }
    OPTIONAL {
        ?trip tro:line ?route .
        ?route tro:abbreviation ?routeName .
        ?route tro:fullTitle ?routeLongName .
    }
}
    """.strip(), """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?trip ?tripShortName ?headsign ?routeName ?routeLongName (MIN(?leaveTime) AS ?departureTime) WHERE {
    ?stopTime a tro:StopEvent .
    ?trip a tro:Trip .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:leaveTime ?leaveTime .
    OPTIONAL { ?trip tro:abbreviation ?tripShortName }
    OPTIONAL { ?trip tro:destination ?headsign }
    OPTIONAL {
        ?trip tro:line ?route .
        ?route a tro:Line .
        ?route tro:abbreviation ?routeName .
        ?route tro:fullTitle ?routeLongName .
    }
}
GROUP BY ?trip ?tripShortName ?headsign ?routeName ?routeLongName
HAVING (MIN(?leaveTime) >= "00:00:00" && MIN(?leaveTime) < "01:00:00")
    """.strip(), """
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?trip ?tripShortName ?headsign ?routeName ?routeLongName (MIN(?leaveTime) AS ?departureTime) WHERE {
    ?stopTime a tro:StopEvent .
    ?stopTime tro:journey ?trip .
    ?stopTime tro:leaveTime ?leaveTime .
    OPTIONAL { ?trip tro:abbreviation ?tripShortName }
    OPTIONAL { ?trip tro:destination ?headsign }
    OPTIONAL {
        ?trip tro:line ?route .
        ?route tro:abbreviation ?routeName .
        ?route tro:fullTitle ?routeLongName .
    }
}
GROUP BY ?trip ?tripShortName ?headsign ?routeName ?routeLongName
HAVING (MIN(?leaveTime) >= "00:00:00" && MIN(?leaveTime) < "01:00:00")
    """.strip()],
    columns=[
        ColumnDef("trip", RelevanceLevel.PREFERRED, "Trip URI", semantic_concept="trip"),
        ColumnDef("tripShortName", RelevanceLevel.PREFERRED, "Trip short name", semantic_concept="trip"),
        ColumnDef("headsign", RelevanceLevel.PREFERRED, "Trip headsign", semantic_concept="trip"),
        ColumnDef("routeName", RelevanceLevel.ACCEPTABLE, "Route short name", semantic_concept="route"),
        ColumnDef("routeLongName", RelevanceLevel.ACCEPTABLE, "Route long name", semantic_concept="route"),
        ColumnDef("departureTime", RelevanceLevel.ACCEPTABLE, "Departure time", semantic_concept="departureTime"),
    ],
    notes="Baseline: Time FILTER. 4 variants: typed/untyped × per-event/per-trip (MIN leaveTime).",
)

# GROUP_CONCAT query - replaces redundant string filter pattern
BASE20 = AdaptiveGroundTruth(
    query_id="BASE20",
    query_set="BASE",
    description="Companies with their operated fields",
    dataset="NRG",
    datasets=["NRG"],
    query="For each active deposit operator, list all the deposits they operate.",
    sparql_queries=[
        # Variant 1: GROUP_CONCAT (aggregated)
        """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?company ?companyName
       (GROUP_CONCAT(?fieldName; separator=", ") AS ?operatedFields)
       (GROUP_CONCAT(STR(?field); separator=", ") AS ?operatedFieldURIs)
       (GROUP_CONCAT(STR(?fieldId); separator=", ") AS ?operatedFieldIds)
WHERE {
    ?field a eno:Deposit .
    ?company a eno:Company .
    ?field eno:activeDepositOperator ?company .
    OPTIONAL { ?field eno:designation ?fieldName }
    OPTIONAL { ?field eno:registryId ?fieldId }
    OPTIONAL { ?company eno:designation ?companyName }
}
GROUP BY ?company ?companyName
        """.strip(),
        # Variant 2: Non-aggregated (one row per company-field pair)
        """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?operator ?operatorName ?deposit ?depositName WHERE {
    ?operator a eno:Company .
    ?deposit a eno:Deposit .
    ?deposit eno:activeDepositOperator ?operator .
    OPTIONAL { ?operator eno:designation ?operatorName }
    OPTIONAL { ?deposit eno:designation ?depositName }
}
ORDER BY ?operatorName ?depositName
        """.strip(),
        # Variant 3: V1 untyped (no implicit ?company type)
        """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?company ?companyName
       (GROUP_CONCAT(?fieldName; separator=", ") AS ?operatedFields)
       (GROUP_CONCAT(STR(?field); separator=", ") AS ?operatedFieldURIs)
       (GROUP_CONCAT(STR(?fieldId); separator=", ") AS ?operatedFieldIds)
WHERE {
    ?field a eno:Deposit .
    ?field eno:activeDepositOperator ?company .
    OPTIONAL { ?field eno:designation ?fieldName }
    OPTIONAL { ?field eno:registryId ?fieldId }
    OPTIONAL { ?company eno:designation ?companyName }
}
GROUP BY ?company ?companyName
        """.strip(),
        # Variant 4: V2 untyped
        """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?operator ?operatorName ?deposit ?depositName WHERE {
    ?deposit eno:activeDepositOperator ?operator .
    OPTIONAL { ?operator eno:designation ?operatorName }
    OPTIONAL { ?deposit eno:designation ?depositName }
}
ORDER BY ?operatorName ?depositName
        """.strip(),
    ],
    columns=[
        [  # Query 1: GROUP_CONCAT aggregation
            ColumnDef("company", RelevanceLevel.PREFERRED, "Company URI", semantic_concept="company"),
            ColumnDef("companyName", RelevanceLevel.PREFERRED, "Company name", semantic_concept="company"),
            ColumnDef("operatedFields", RelevanceLevel.PREFERRED, "Operated fields (names)", semantic_concept="fields"),
            ColumnDef("operatedFieldURIs", RelevanceLevel.ACCEPTABLE, "Operated field URIs", semantic_concept="fields"),
            ColumnDef("operatedFieldIds", RelevanceLevel.ACCEPTABLE, "Operated field IDs", semantic_concept="fields"),
        ],
        [  # Query 2: Non-aggregated rows
            ColumnDef("operator", RelevanceLevel.PREFERRED, "Operator URI", semantic_concept="company"),
            ColumnDef("operatorName", RelevanceLevel.PREFERRED, "Operator name", semantic_concept="company"),
            ColumnDef("deposit", RelevanceLevel.PREFERRED, "Deposit URI", semantic_concept="field"),
            ColumnDef("depositName", RelevanceLevel.PREFERRED, "Deposit name", semantic_concept="field"),
        ],
        [  # Query 3: V1 untyped (same columns as V1)
            ColumnDef("company", RelevanceLevel.PREFERRED, "Company URI", semantic_concept="company"),
            ColumnDef("companyName", RelevanceLevel.PREFERRED, "Company name", semantic_concept="company"),
            ColumnDef("operatedFields", RelevanceLevel.PREFERRED, "Operated fields (names)", semantic_concept="fields"),
            ColumnDef("operatedFieldURIs", RelevanceLevel.ACCEPTABLE, "Operated field URIs", semantic_concept="fields"),
            ColumnDef("operatedFieldIds", RelevanceLevel.ACCEPTABLE, "Operated field IDs", semantic_concept="fields"),
        ],
        [  # Query 4: V2 untyped (same columns as V2)
            ColumnDef("operator", RelevanceLevel.PREFERRED, "Operator URI", semantic_concept="company"),
            ColumnDef("operatorName", RelevanceLevel.PREFERRED, "Operator name", semantic_concept="company"),
            ColumnDef("deposit", RelevanceLevel.PREFERRED, "Deposit URI", semantic_concept="field"),
            ColumnDef("depositName", RelevanceLevel.PREFERRED, "Deposit name", semantic_concept="field"),
        ],
    ],
    notes="Baseline: Four variants: typed/untyped × aggregated/non-aggregated.",
)

# Det norske oljeselskap - company with known synonym (Aker BP)
BASE21 = AdaptiveGroundTruth(
    query_id="BASE21",
    query_set="BASE",
    description="Fields operated by Det norske oljeselskap",
    dataset="NRG",
    datasets=["NRG"],
    query="Which deposits are operated by Det norske oljeselskap ASA?",
    sparql_queries=["""
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?field ?fieldName WHERE {
    ?field a eno:Deposit .
    ?company a eno:Company .
    ?field eno:activeDepositOperator ?company .
    ?company eno:designation ?companyName .
    FILTER(REGEX(?companyName, "Det norske oljeselskap", "i"))
    OPTIONAL { ?field eno:designation ?fieldName }
}
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?field ?fieldName WHERE {
    ?field a eno:Deposit .
    ?field eno:activeDepositOperator ?company .
    ?company eno:designation ?companyName .
    FILTER(REGEX(?companyName, "Det norske oljeselskap", "i"))
    OPTIONAL { ?field eno:designation ?fieldName }
}
    """.strip()],
    columns=[
        ColumnDef("field", RelevanceLevel.PREFERRED, "Field URI", semantic_concept="field"),
        ColumnDef("fieldName", RelevanceLevel.PREFERRED, "Field name", semantic_concept="field"),
    ],
    notes="Baseline: Named entity. Det norske merged with BP to form Aker BP in 2016.",
)

# Collect all BASE queries (17 total after removing BASE02, BASE07, BASE15, BASE17)
SET_BASE = [
    BASE01, BASE03, BASE04, BASE05, BASE06, BASE08,
    BASE09, BASE10, BASE11, BASE12, BASE13, BASE14, BASE16,
    BASE18, BASE19, BASE20, BASE21,
]


# =============================================================================
# SET SYN: Natural Language Variants (16 total)
# Same SPARQL/GT as BASE, everyday language instead of mapping terms
# EDU/TRN/NRG: uses common terms (students, routes, fields, etc.)
# =============================================================================

def _create_synonym_variant(
    base: AdaptiveGroundTruth,
    new_query: str,
    description: str,
    extra_sparql_queries: list[str] | None = None,
    extra_notes: str = "",
) -> AdaptiveGroundTruth:
    """Create a synonym variant that shares GT with base query.

    extra_sparql_queries: additional admissible readings that only apply to the SYN wording
    (appended after the base readings; the evaluator scores every reading and keeps the best).
    """
    return AdaptiveGroundTruth(
        query_id=f"SYN{base.query_id[4:]}",  # BASE01 -> SYN01
        query_set="SYN",
        description=f"{base.description} (synonym)",
        dataset=base.dataset,
        datasets=base.datasets,
        query=new_query,
        sparql_queries=list(base.sparql_queries) + list(extra_sparql_queries or []),
        columns=base.columns,  # Same columns
        notes=f"Synonym variant of {base.query_id}: {description}" + (f" {extra_notes}" if extra_notes else ""),
    )


SYN01 = _create_synonym_variant(
    BASE01,
    "Show me 5000 cases of students taking classes taught by professors from their own department.",
    "learners->students, courses->classes, enrolled in->taking, instructed by->taught by, educators->professors, division->department, examples->cases",
)

SYN03 = _create_synonym_variant(
    BASE03,
    "Show unit processes with their exchanges.",
    "list->show, activities->unit processes, flows->exchanges (LCA terminology)",
)

SYN04 = _create_synonym_variant(
    BASE04,
    "Show me 1000 cases of which transit routes pass through which platforms.",
    "lines->transit routes, stop events at->pass through, stations->platforms, examples->cases (paraphrase)",
)

SYN05 = _create_synonym_variant(
    BASE05,
    "Per department, how many people are registered for classes, and which ones?",
    "learners->people (hypernym), division->department, enrolled in->registered for, courses->classes, restructured",
)

SYN06 = _create_synonym_variant(
    BASE06,
    "For each transit route, how many times does it halt altogether and what is the mean per trip?",
    "line->transit route, stop events->halt (circumlocution), journey->trip, total->altogether, average->mean",
)

SYN08 = _create_synonym_variant(
    BASE08,
    "Which 10 places in the transit network are served by the most trips?",
    "stations->places (hypernym), journeys->trips, 'transit network' for domain context, 'served by' circumlocution",
)

SYN09 = _create_synonym_variant(
    BASE09,
    "Which faculty teaching upper-level classes have nobody under their supervision?",
    "educators->faculty (hypernym), instruct->teaching, advanced modules->upper-level classes, mentees->nobody under supervision (circumlocution)",
)

SYN10 = _create_synonym_variant(
    BASE10,
    "Which items are the least expensive and the costliest?",
    "what->which, products->items, cheapest->least expensive, most expensive->costliest",
)

SYN11 = _create_synonym_variant(
    BASE11,
    "Where has more than 50000000 in oil been pumped out altogether?",
    "deposits->where (implicit), extracted crude->oil pumped out (informal+passive), 50 million->50000000 (numeric), total->altogether",
)

SYN12 = _create_synonym_variant(
    BASE12,
    "Show every undergraduate and graduate class alongside who teaches it and which department it falls under.",
    "list->show, basic modules->undergraduate class, advanced modules->graduate class, instructors->who teaches it (circumlocution), division->department, restructured",
)

SYN13 = _create_synonym_variant(
    BASE13,
    "Show the status and managing company of the Troll oil field.",
    "condition->status, active deposit operator->managing company, deposit->oil field, TROLL->Troll",
)

SYN14 = _create_synonym_variant(
    BASE14,
    "Show classes offered at Uni 0.",
    "courses->classes, instructed->offered, University0->Uni 0",
)

SYN16 = _create_synonym_variant(
    BASE16,
    "Which fields are currently operated by Equinor?",
    "deposits->fields, active deposit operator->currently operated by, Statoil->Equinor",
)

SYN18 = _create_synonym_variant(
    BASE18,
    "Which drill holes go deeper than 3 kilometers?",
    "boreholes->drill holes, total drill depth greater than->go deeper than, 3000 meters->3 kilometers (unit change)",
)

SYN19 = _create_synonym_variant(
    BASE19,
    "Show transit trips departing between midnight and 1 AM.",
    "journeys->transit trips, leave times->departing, 00:00:00->midnight, 01:00:00->1 AM (time format change)",
)

SYN20 = _create_synonym_variant(
    BASE20,
    "For each oil company, which production sites do they currently manage?",
    "active deposit operator->oil company, deposits->production sites, operate->currently manage, list->which (circumlocution)",
)

SYN21 = _create_synonym_variant(
    BASE21,
    "Which fields does Aker BP currently manage?",
    "deposits->fields, operated->currently manage, Det norske oljeselskap ASA->Aker BP",
    extra_sparql_queries=["""
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?field ?fieldName WHERE {
    ?field a eno:Deposit .
    ?company a eno:Company .
    ?field eno:activeDepositOperator ?company .
    ?company eno:designation ?companyName .
    FILTER(REGEX(?companyName, "Det norske oljeselskap|BP Norge", "i"))
    OPTIONAL { ?field eno:designation ?fieldName }
}
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?field ?fieldName WHERE {
    ?field a eno:Deposit .
    ?field eno:activeDepositOperator ?company .
    ?company eno:designation ?companyName .
    FILTER(REGEX(?companyName, "Det norske oljeselskap|BP Norge", "i"))
    OPTIONAL { ?field eno:designation ?fieldName }
}
    """.strip()],
    extra_notes=(
        "Readings 3 and 4 added on 2026-09-15: Aker BP was formed in 2016 by merging Det norske "
        "oljeselskap ASA with BP Norge AS. The data still records both predecessors as active "
        "operators (Det norske: 2 fields, BP Norge: 6 fields) and lists 'Aker BP ASA' as a company "
        "without operated fields, so the fields of both predecessors are an admissible answer to the "
        "SYN wording. Added after inspecting the stage-2 runs; BASE21 keeps the Det norske-only readings."
    ),
)

# Collect all SYN queries (17 total after removing SYN02, SYN07, SYN15, SYN17)
SET_SYN = [
    SYN01, SYN03, SYN04, SYN05, SYN06, SYN08,
    SYN09, SYN10, SYN11, SYN12, SYN13, SYN14, SYN16,
    SYN18, SYN19, SYN20, SYN21,
]


# =============================================================================
# SET TYPO: Typo Variants (16 total)
# Same SPARQL/GT as BASE, mapping terminology with spelling errors
# Tests fuzzy matching and error tolerance on ontology terms
# =============================================================================

def _create_typo_variant(base: AdaptiveGroundTruth, new_query: str, description: str) -> AdaptiveGroundTruth:
    """Create a typo variant that shares GT with base query."""
    return AdaptiveGroundTruth(
        query_id=f"TYPO{base.query_id[4:]}",  # BASE01 -> TYPO01
        query_set="TYPO",
        description=f"{base.description} (typo)",
        dataset=base.dataset,
        datasets=base.datasets,
        query=new_query,
        sparql_queries=base.sparql_queries,  # Same SPARQL
        columns=base.columns,  # Same columns
        notes=f"Typo variant of {base.query_id}: {description}",
    )


TYPO01 = _create_typo_variant(
    BASE01,
    "Show me 5000 exampels of learnres enrolld in coureses instructd by educatros employd at their own divison.",
    "examples->exampels, learners->learnres, enrolled->enrolld, courses->coureses, instructed->instructd, educators->educatros, employed->employd, division->divison",
)

TYPO03 = _create_typo_variant(
    BASE03,
    "List activites with thier input and outptu flows.",
    "activities->activites, their->thier, output->outptu",
)

TYPO04 = _create_typo_variant(
    BASE04,
    "Show me 1000 exampels of wich lines have stopp events at wich statinos.",
    "examples->exampels, which->wich (x2), stop->stopp, stations->statinos",
)

TYPO05 = _create_typo_variant(
    BASE05,
    "How manny learnres in each divison are enrolld in coureses, and what coureses are they enrolld in?",
    "many->manny, learners->learnres, division->divison, enrolled->enrolld (x2), courses->coureses (x2)",
)

TYPO06 = _create_typo_variant(
    BASE06,
    "What is the totel number of stopp events and averge stopp events per journy for each lin?",
    "total->totel, stop->stopp (x2), average->averge, journey->journy, line->lin",
)

TYPO08 = _create_typo_variant(
    BASE08,
    "Show the top 10 statinos with the mostt journyes.",
    "stations->statinos, most->mostt, journeys->journyes",
)

TYPO09 = _create_typo_variant(
    BASE09,
    "Which educatros who instrcut advnaced moduels have no menteees?",
    "educators->educatros, instruct->instrcut, advanced->advnaced, modules->moduels, mentees->menteees",
)

TYPO10 = _create_typo_variant(
    BASE10,
    "What are the cheepest and most expensiv prodcuts?",
    "cheapest->cheepest, expensive->expensiv, products->prodcuts",
)

TYPO11 = _create_typo_variant(
    BASE11,
    "Which deposists have more than fivty million in extrcated crude totel?",
    "deposits->deposists, fifty->fivty, extracted->extrcated, total->totel",
)

TYPO12 = _create_typo_variant(
    BASE12,
    "List all baisc moduels and advnaced moduels with their instructros and divison.",
    "basic->baisc, modules->moduels (x2), advanced->advnaced, instructors->instructros, division->divison",
)

TYPO13 = _create_typo_variant(
    BASE13,
    "Show the conditon and actve deposit opertor of the TROL deposite.",
    "condition->conditon, active->actve, operator->opertor, TROLL->TROL, deposit->deposite",
)

TYPO14 = _create_typo_variant(
    BASE14,
    "Show coureses instructd at Univeristy0.",
    "courses->coureses, instructed->instructd, University->Univeristy",
)

TYPO16 = _create_typo_variant(
    BASE16,
    "Which deposists have Statoil as their actve deposit opertor?",
    "deposits->deposists, active->actve, operator->opertor",
)

TYPO18 = _create_typo_variant(
    BASE18,
    "Find boreholse with a totel drill deptth greater than 3000 meeters.",
    "boreholes->boreholse, total->totel, depth->deptth, meters->meeters",
)

TYPO19 = _create_typo_variant(
    BASE19,
    "Show journyes with leav tiems between 00:00:000 and 01:00:000.",
    "journeys->journyes, leave->leav, times->tiems, 00:00:00->00:00:000, 01:00:00->01:00:000",
)

TYPO20 = _create_typo_variant(
    BASE20,
    "For each actve deposit opertor, lsit all the deposists they oprate.",
    "active->actve, operator->opertor, list->lsit, deposits->deposists, operate->oprate",
)

TYPO21 = _create_typo_variant(
    BASE21,
    "Which deposists are operatd by Det norske oljeslskap ASA?",
    "deposits->deposists, operated->operatd, oljeselskap->oljeslskap",
)

# Collect all TYPO queries (17 total after removing TYPO02, TYPO07, TYPO15, TYPO17)
SET_TYPO = [
    TYPO01, TYPO03, TYPO04, TYPO05, TYPO06, TYPO08,
    TYPO09, TYPO10, TYPO11, TYPO12, TYPO13, TYPO14, TYPO16,
    TYPO18, TYPO19, TYPO20, TYPO21,
]


# =============================================================================
# SET UNDER: Underspecified Queries (5 total)
# Vague terms requiring interpretation - diverse underspecification types:
# - Ranking: "top" (UNDER01)
# - Temporal: "recent" (UNDER03)
# - Size: "large" (UNDER04)
# - Domain-specific: "significant" in LCA (UNDER06)
# - Threshold: "affordable" (UNDER08)
# =============================================================================

UNDER01 = AdaptiveGroundTruth(
    query_id="UNDER01",
    query_set="UNDER",
    description="Top products",
    dataset="BSBM",
    datasets=["BSBM"],
    query="Show top products.",
    sparql_queries=["""
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?product ?label (AVG(?rating) AS ?avgRating) WHERE {
    ?product a bsbm:Product .
    ?review a bsbm:Review .
    ?review bsbm:reviewFor ?product .
    ?review bsbm:rating1 ?rating .
    OPTIONAL { ?product rdfs:label ?label }
}
GROUP BY ?product ?label
ORDER BY DESC(?avgRating)
    """.strip(), """
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?product ?label (AVG(?rating) AS ?avgRating) WHERE {
    ?product a bsbm:Product .
    ?review bsbm:reviewFor ?product .
    ?review bsbm:rating1 ?rating .
    OPTIONAL { ?product rdfs:label ?label }
}
GROUP BY ?product ?label
ORDER BY DESC(?avgRating)
    """.strip()],
    columns=[
        ColumnDef("product", RelevanceLevel.PREFERRED, "Product URI", semantic_concept="product"),
        ColumnDef("label", RelevanceLevel.PREFERRED, "Product label", semantic_concept="product"),
        ColumnDef("avgRating", RelevanceLevel.ACCEPTABLE, "Average rating", is_measurement=True, semantic_concept="top"),
    ],
    requires_prefix_match=True,
    notes="Underspecified: 'top' = how many? by what metric?",
)

# UNDER02 removed: "popular" redundant with "top" (both ranking-based ambiguity)

UNDER03 = AdaptiveGroundTruth(
    query_id="UNDER03",
    query_set="UNDER",
    description="Recent production data",
    dataset="NRG",
    datasets=["NRG"],
    query="Get recent production data.",
    sparql_queries=["""
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?field ?fieldName ?year ?oil ?gas WHERE {
    ?prod a eno:AnnualDepositOutput .
    ?field a eno:Deposit .
    ?prod eno:outputForDeposit ?field .
    ?prod eno:outputYear ?year .
    FILTER(?year >= 2010)
    OPTIONAL { ?prod eno:extractedCrude ?oil }
    OPTIONAL { ?prod eno:extractedGas ?gas }
    OPTIONAL { ?field eno:designation ?fieldName }
}
ORDER BY DESC(?year)
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?field ?fieldName ?year ?oil ?gas WHERE {
    ?prod a eno:AnnualDepositOutput .
    ?prod eno:outputForDeposit ?field .
    ?prod eno:outputYear ?year .
    FILTER(?year >= 2010)
    OPTIONAL { ?prod eno:extractedCrude ?oil }
    OPTIONAL { ?prod eno:extractedGas ?gas }
    OPTIONAL { ?field eno:designation ?fieldName }
}
ORDER BY DESC(?year)
    """.strip()],
    columns=[
        ColumnDef("field", RelevanceLevel.PREFERRED, "Field URI", semantic_concept="field"),
        ColumnDef("fieldName", RelevanceLevel.PREFERRED, "Field name", semantic_concept="field"),
        ColumnDef("year", RelevanceLevel.PREFERRED, "Year", semantic_concept="recent"),
        ColumnDef("oil", RelevanceLevel.PREFERRED, "Oil production", semantic_concept="productionData"),
        ColumnDef("gas", RelevanceLevel.PREFERRED, "Gas production", semantic_concept="productionData"),
    ],
    notes="Underspecified: 'recent' = what year threshold?",
)

UNDER04 = AdaptiveGroundTruth(
    query_id="UNDER04",
    query_set="UNDER",
    description="Large departments",
    dataset="EDU",
    datasets=["EDU"],
    query="Find large departments.",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
SELECT ?dept ?deptName (COUNT(?faculty) AS ?facultyCount) WHERE {
    ?dept a eduo:Division .
    ?faculty a eduo:Educator .
    ?faculty eduo:employedAt ?dept .
    OPTIONAL { ?dept eduo:label ?deptName }
}
GROUP BY ?dept ?deptName
ORDER BY DESC(?facultyCount)
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
SELECT ?dept ?deptName (COUNT(?faculty) AS ?facultyCount) WHERE {
    ?dept a eduo:Division .
    ?faculty eduo:employedAt ?dept .
    OPTIONAL { ?dept eduo:label ?deptName }
}
GROUP BY ?dept ?deptName
ORDER BY DESC(?facultyCount)
    """.strip()],
    columns=[
        ColumnDef("dept", RelevanceLevel.PREFERRED, "Department URI", semantic_concept="department"),
        ColumnDef("deptName", RelevanceLevel.PREFERRED, "Department name", semantic_concept="department"),
        ColumnDef("facultyCount", RelevanceLevel.ACCEPTABLE, "Faculty count", is_measurement=True, semantic_concept="large"),
    ],
    requires_prefix_match=True,
    notes="Underspecified: 'large' = by faculty? students? budget?",
)

# UNDER05 removed: "major" redundant with "large" (both size-based ambiguity)

# LCA dataset query - underspecified "significant" emissions
UNDER06 = AdaptiveGroundTruth(
    query_id="UNDER06",
    query_set="UNDER",
    description="Significant emissions",
    dataset="LCA",
    datasets=["LCA"],
    query="Find significant emissions.",
    sparql_queries=["""
PREFIX oriont: <https://orienting.eu/oriont#>
PREFIX oriont-ext: <https://orienting.eu/oriont-ext#>
SELECT ?flow ?flowName ?value WHERE {
    ?flow a oriont:ElementaryFlow .
    
    ?flow oriont-ext:hasFlowName ?flowName .
    ?flow oriont-ext:hasMeasureValue ?value .
}
    """.strip()],
    columns=[
        ColumnDef("flow", RelevanceLevel.PREFERRED, "Flow URI", semantic_concept="emission"),
        ColumnDef("flowName", RelevanceLevel.PREFERRED, "Flow name", semantic_concept="emission"),
        ColumnDef("value", RelevanceLevel.PREFERRED, "Measure value", is_measurement=True, semantic_concept="significant"),
    ],
    requires_prefix_match=True,
    notes="Underspecified: 'significant' = by measure value? by CF impact? by category? LCA dataset.",
)

# UNDER07 removed: "prolific" redundant with "top" (both ranking-based ambiguity)

UNDER08 = AdaptiveGroundTruth(
    query_id="UNDER08",
    query_set="UNDER",
    description="Affordable products",
    dataset="BSBM",
    datasets=["BSBM"],
    query="Show affordable products.",
    sparql_queries=["""
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?product ?label ?price WHERE {
    ?product a bsbm:Product .
    ?offer a bsbm:Offer .
    ?offer bsbm:product ?product .
    ?offer bsbm:price ?price .
    FILTER(?price < 100)
    OPTIONAL { ?product rdfs:label ?label }
}
ORDER BY ?price
    """.strip(), """
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?product ?label ?price WHERE {
    ?product a bsbm:Product .
    ?offer bsbm:product ?product .
    ?offer bsbm:price ?price .
    FILTER(?price < 100)
    OPTIONAL { ?product rdfs:label ?label }
}
ORDER BY ?price
    """.strip()],
    columns=[
        ColumnDef("product", RelevanceLevel.PREFERRED, "Product URI", semantic_concept="product"),
        ColumnDef("label", RelevanceLevel.PREFERRED, "Product label", semantic_concept="product"),
        ColumnDef("price", RelevanceLevel.ACCEPTABLE, "Price", semantic_concept="affordable"),
    ],
    notes="Underspecified: 'affordable' = what price threshold?",
)

# UNDER09 removed: "experienced" redundant with "large" (both subjective adjective for EDU entities)
# UNDER10 removed: "significant production" redundant with UNDER06 "significant emissions"

# Collect all UNDER queries (5 diverse underspecification types)
SET_UNDER = [
    UNDER01,  # "top" - ranking ambiguity (by what metric?)
    UNDER03,  # "recent" - temporal ambiguity (what time threshold?)
    UNDER04,  # "large" - size ambiguity (by what measure?)
    UNDER06,  # "significant" - domain-specific ambiguity (LCA dataset)
    UNDER08,  # "affordable" - price threshold ambiguity
]


# =============================================================================
# SET CROSS: Cross-Dataset Queries (5 total)
# Multi-endpoint queries - NO synonym/typo/semantic ambiguities
# Uses UNION pattern: Single query sent to all endpoints, results merged.
# =============================================================================

# CROSS01: Entity names from different domains (not just counts)
CROSS01 = AdaptiveGroundTruth(
    query_id="CROSS01",
    query_set="CROSS",
    description="University and oil field names",
    dataset="EDU",
    datasets=["EDU", "NRG"],
    query="List all university names and all oil field names.",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?name ?sourceType WHERE {
    {
        ?uni a eduo:Academy .
        ?uni eduo:label ?name .
        BIND("university" AS ?sourceType)
    }
    UNION
    {
        ?field a eno:Deposit .
        ?field eno:designation ?name .
        BIND("oilfield" AS ?sourceType)
    }
}
ORDER BY ?sourceType ?name
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?universityName ?fieldName WHERE {
    {
        ?uni a eduo:Academy .
        ?uni eduo:label ?universityName .
    }
    UNION
    {
        ?field a eno:Deposit .
        ?field eno:designation ?fieldName .
    }
}
    """.strip()],
    columns=[
        [  # Query 1: normalized ?name + ?sourceType
            ColumnDef("name", RelevanceLevel.PREFERRED, "Entity name", semantic_concept="name"),
            ColumnDef("sourceType", RelevanceLevel.ACCEPTABLE, "Source type", semantic_concept="sourceType", is_bind_label=True),
        ],
        [  # Query 2: domain-specific ?universityName + ?fieldName
            ColumnDef("universityName", RelevanceLevel.PREFERRED, "University name", semantic_concept="universityName"),
            ColumnDef("fieldName", RelevanceLevel.PREFERRED, "Field name", semantic_concept="fieldName"),
        ],
    ],
    notes="Cross-dataset: EDU + NRG. Alt: UNION with domain-specific variable names.",
)

# CROSS02: Aggregations with AVG (more complex than simple COUNT)
CROSS02 = AdaptiveGroundTruth(
    query_id="CROSS02",
    query_set="CROSS",
    description="Average courses per department and stops per route",
    dataset="EDU",
    datasets=["EDU", "TRN"],
    query="What is the average number of courses per department and the average number of stops per route?",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?category (AVG(?itemCount) AS ?average) WHERE {
    {
        SELECT ?dept (COUNT(?course) AS ?itemCount) ("courses_per_department" AS ?category) WHERE {
            ?dept a eduo:Division .
            ?prof a eduo:Educator .
            ?course a eduo:Course .
            ?prof eduo:employedAt ?dept .
            ?prof eduo:instructorOf ?course .
        }
        GROUP BY ?dept
    }
    UNION
    {
        SELECT ?route (COUNT(DISTINCT ?stop) AS ?itemCount) ("stops_per_route" AS ?category) WHERE {
            ?route a tro:Line .
            ?trip a tro:Trip .
            ?stopTime a tro:StopEvent .
            ?stop a tro:Stop .
            ?trip tro:line ?route .
            ?stopTime tro:journey ?trip .
            ?stopTime tro:station ?stop .
        }
        GROUP BY ?route
    }
}
GROUP BY ?category
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?category (AVG(?itemCount) AS ?average) WHERE {
    {
        SELECT ?dept (COUNT(?course) AS ?itemCount) ("courses_per_department" AS ?category) WHERE {
            ?dept a eduo:Division .
            ?prof eduo:employedAt ?dept .
            ?prof eduo:instructorOf ?course .
        }
        GROUP BY ?dept
    }
    UNION
    {
        SELECT ?route (COUNT(DISTINCT ?stop) AS ?itemCount) ("stops_per_route" AS ?category) WHERE {
            ?route a tro:Line .
            ?stopTime a tro:StopEvent .
            ?trip tro:line ?route .
            ?stopTime tro:journey ?trip .
            ?stopTime tro:station ?stop .
        }
        GROUP BY ?route
    }
}
GROUP BY ?category
    """.strip()],
    columns=[
        ColumnDef("category", RelevanceLevel.ACCEPTABLE, "Metric category", semantic_concept="category", is_bind_label=True),
        ColumnDef("average", RelevanceLevel.PREFERRED, "Average value", is_measurement=True, semantic_concept="average"),
    ],
    notes="Cross-dataset: EDU + TRN. Complex AVG aggregation over subqueries.",
)

# CROSS03: Top-N rankings from different domains
CROSS03 = AdaptiveGroundTruth(
    query_id="CROSS03",
    query_set="CROSS",
    description="Top 2 departments by faculty and top 2 stops by trips",
    dataset="EDU",
    datasets=["EDU", "TRN"],
    query="Show the top 2 largest departments by faculty count and the top 2 busiest stops by trip count.",
    sparql_queries=[
        # Variant 1: normalized, COUNT without DISTINCT
        """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?name ?count ?rankType WHERE {
    {
        SELECT ?name ?count ("top_department" AS ?rankType) WHERE {
            SELECT ?dept (SAMPLE(?deptName) AS ?name) (COUNT(?faculty) AS ?count) WHERE {
                ?dept a eduo:Division .
                ?faculty a eduo:Educator .
                ?faculty eduo:employedAt ?dept .
                OPTIONAL { ?dept eduo:label ?deptName }
            }
            GROUP BY ?dept
            ORDER BY DESC(?count)
            LIMIT 2
        }
    }
    UNION
    {
        SELECT ?name ?count ("top_stop" AS ?rankType) WHERE {
            SELECT ?stop (SAMPLE(?stopName) AS ?name) (COUNT(?trip) AS ?count) WHERE {
                ?stopTime a tro:StopEvent .
                ?stop a tro:Stop .
                ?trip a tro:Trip .
                ?stopTime tro:station ?stop .
                ?stopTime tro:journey ?trip .
                OPTIONAL { ?stop foaf:name ?stopName }
            }
            GROUP BY ?stop
            ORDER BY DESC(?count)
            LIMIT 2
        }
    }
}
ORDER BY ?rankType DESC(?count)
        """.strip(),
        # Variant 2: domain-specific, COUNT without DISTINCT
        """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?department ?facultyCount ?stop ?tripCount WHERE {
    {
        SELECT ?department (COUNT(?faculty) AS ?facultyCount) WHERE {
            ?department a eduo:Division .
            ?faculty a eduo:Educator .
            ?faculty eduo:employedAt ?department .
        }
        GROUP BY ?department
        ORDER BY DESC(?facultyCount)
        LIMIT 2
    }
    UNION
    {
        SELECT ?stop (COUNT(?trip) AS ?tripCount) WHERE {
            ?stopTime a tro:StopEvent .
            ?stop a tro:Stop .
            ?trip a tro:Trip .
            ?stopTime tro:station ?stop .
            ?stopTime tro:journey ?trip .
        }
        GROUP BY ?stop
        ORDER BY DESC(?tripCount)
        LIMIT 2
    }
}
        """.strip(),
        # Variant 3: normalized, COUNT DISTINCT trips (some stops have multiple StopTimes per trip)
        """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?name ?count ?rankType WHERE {
    {
        SELECT ?name ?count ("top_department" AS ?rankType) WHERE {
            SELECT ?dept (SAMPLE(?deptName) AS ?name) (COUNT(?faculty) AS ?count) WHERE {
                ?dept a eduo:Division .
                ?faculty a eduo:Educator .
                ?faculty eduo:employedAt ?dept .
                OPTIONAL { ?dept eduo:label ?deptName }
            }
            GROUP BY ?dept
            ORDER BY DESC(?count)
            LIMIT 2
        }
    }
    UNION
    {
        SELECT ?name ?count ("top_stop" AS ?rankType) WHERE {
            SELECT ?stop (SAMPLE(?stopName) AS ?name) (COUNT(DISTINCT ?trip) AS ?count) WHERE {
                ?stopTime a tro:StopEvent .
                ?stop a tro:Stop .
                ?trip a tro:Trip .
                ?stopTime tro:station ?stop .
                ?stopTime tro:journey ?trip .
                OPTIONAL { ?stop foaf:name ?stopName }
            }
            GROUP BY ?stop
            ORDER BY DESC(?count)
            LIMIT 2
        }
    }
}
ORDER BY ?rankType DESC(?count)
        """.strip(),
        # Variant 4: domain-specific, COUNT DISTINCT trips
        """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?department ?facultyCount ?stop ?tripCount WHERE {
    {
        SELECT ?department (COUNT(?faculty) AS ?facultyCount) WHERE {
            ?department a eduo:Division .
            ?faculty a eduo:Educator .
            ?faculty eduo:employedAt ?department .
        }
        GROUP BY ?department
        ORDER BY DESC(?facultyCount)
        LIMIT 2
    }
    UNION
    {
        SELECT ?stop (COUNT(DISTINCT ?trip) AS ?tripCount) WHERE {
            ?stopTime a tro:StopEvent .
            ?stop a tro:Stop .
            ?trip a tro:Trip .
            ?stopTime tro:station ?stop .
            ?stopTime tro:journey ?trip .
        }
        GROUP BY ?stop
        ORDER BY DESC(?tripCount)
        LIMIT 2
    }
}
        """.strip(),
        # Variant 5: V1 untyped (no ?faculty, ?stop, ?trip types)
        """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?name ?count ?rankType WHERE {
    {
        SELECT ?name ?count ("top_department" AS ?rankType) WHERE {
            SELECT ?dept (SAMPLE(?deptName) AS ?name) (COUNT(?faculty) AS ?count) WHERE {
                ?dept a eduo:Division .
                ?faculty eduo:employedAt ?dept .
                OPTIONAL { ?dept eduo:label ?deptName }
            }
            GROUP BY ?dept
            ORDER BY DESC(?count)
            LIMIT 2
        }
    }
    UNION
    {
        SELECT ?name ?count ("top_stop" AS ?rankType) WHERE {
            SELECT ?stop (SAMPLE(?stopName) AS ?name) (COUNT(?trip) AS ?count) WHERE {
                ?stopTime a tro:StopEvent .
                ?stopTime tro:station ?stop .
                ?stopTime tro:journey ?trip .
                OPTIONAL { ?stop foaf:name ?stopName }
            }
            GROUP BY ?stop
            ORDER BY DESC(?count)
            LIMIT 2
        }
    }
}
ORDER BY ?rankType DESC(?count)
        """.strip(),
        # Variant 6: V2 untyped
        """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?department ?facultyCount ?stop ?tripCount WHERE {
    {
        SELECT ?department (COUNT(?faculty) AS ?facultyCount) WHERE {
            ?department a eduo:Division .
            ?faculty eduo:employedAt ?department .
        }
        GROUP BY ?department
        ORDER BY DESC(?facultyCount)
        LIMIT 2
    }
    UNION
    {
        SELECT ?stop (COUNT(?trip) AS ?tripCount) WHERE {
            ?stopTime a tro:StopEvent .
            ?stopTime tro:station ?stop .
            ?stopTime tro:journey ?trip .
        }
        GROUP BY ?stop
        ORDER BY DESC(?tripCount)
        LIMIT 2
    }
}
        """.strip(),
        # Variant 7: V3 untyped (COUNT DISTINCT)
        """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?name ?count ?rankType WHERE {
    {
        SELECT ?name ?count ("top_department" AS ?rankType) WHERE {
            SELECT ?dept (SAMPLE(?deptName) AS ?name) (COUNT(?faculty) AS ?count) WHERE {
                ?dept a eduo:Division .
                ?faculty eduo:employedAt ?dept .
                OPTIONAL { ?dept eduo:label ?deptName }
            }
            GROUP BY ?dept
            ORDER BY DESC(?count)
            LIMIT 2
        }
    }
    UNION
    {
        SELECT ?name ?count ("top_stop" AS ?rankType) WHERE {
            SELECT ?stop (SAMPLE(?stopName) AS ?name) (COUNT(DISTINCT ?trip) AS ?count) WHERE {
                ?stopTime a tro:StopEvent .
                ?stopTime tro:station ?stop .
                ?stopTime tro:journey ?trip .
                OPTIONAL { ?stop foaf:name ?stopName }
            }
            GROUP BY ?stop
            ORDER BY DESC(?count)
            LIMIT 2
        }
    }
}
ORDER BY ?rankType DESC(?count)
        """.strip(),
        # Variant 8: V4 untyped (domain-specific, COUNT DISTINCT)
        """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?department ?facultyCount ?stop ?tripCount WHERE {
    {
        SELECT ?department (COUNT(?faculty) AS ?facultyCount) WHERE {
            ?department a eduo:Division .
            ?faculty eduo:employedAt ?department .
        }
        GROUP BY ?department
        ORDER BY DESC(?facultyCount)
        LIMIT 2
    }
    UNION
    {
        SELECT ?stop (COUNT(DISTINCT ?trip) AS ?tripCount) WHERE {
            ?stopTime a tro:StopEvent .
            ?stopTime tro:station ?stop .
            ?stopTime tro:journey ?trip .
        }
        GROUP BY ?stop
        ORDER BY DESC(?tripCount)
        LIMIT 2
    }
}
        """.strip(),
    ],
    columns=[
        [  # V1: normalized, COUNT
            ColumnDef("name", RelevanceLevel.PREFERRED, "Entity name", semantic_concept="name"),
            ColumnDef("count", RelevanceLevel.PREFERRED, "Count", is_measurement=True, semantic_concept="count"),
            ColumnDef("rankType", RelevanceLevel.ACCEPTABLE, "Ranking category", semantic_concept="rankType", is_bind_label=True),
        ],
        [  # V2: domain-specific, COUNT
            ColumnDef("department", RelevanceLevel.PREFERRED, "Department URI", semantic_concept="department"),
            ColumnDef("facultyCount", RelevanceLevel.PREFERRED, "Faculty count", is_measurement=True, semantic_concept="facultyCount"),
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.PREFERRED, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # V3: normalized, COUNT DISTINCT
            ColumnDef("name", RelevanceLevel.PREFERRED, "Entity name", semantic_concept="name"),
            ColumnDef("count", RelevanceLevel.PREFERRED, "Count", is_measurement=True, semantic_concept="count"),
            ColumnDef("rankType", RelevanceLevel.ACCEPTABLE, "Ranking category", semantic_concept="rankType", is_bind_label=True),
        ],
        [  # V4: domain-specific, COUNT DISTINCT
            ColumnDef("department", RelevanceLevel.PREFERRED, "Department URI", semantic_concept="department"),
            ColumnDef("facultyCount", RelevanceLevel.PREFERRED, "Faculty count", is_measurement=True, semantic_concept="facultyCount"),
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.PREFERRED, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # V5: V1 untyped
            ColumnDef("name", RelevanceLevel.PREFERRED, "Entity name", semantic_concept="name"),
            ColumnDef("count", RelevanceLevel.PREFERRED, "Count", is_measurement=True, semantic_concept="count"),
            ColumnDef("rankType", RelevanceLevel.ACCEPTABLE, "Ranking category", semantic_concept="rankType", is_bind_label=True),
        ],
        [  # V6: V2 untyped
            ColumnDef("department", RelevanceLevel.PREFERRED, "Department URI", semantic_concept="department"),
            ColumnDef("facultyCount", RelevanceLevel.PREFERRED, "Faculty count", is_measurement=True, semantic_concept="facultyCount"),
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.PREFERRED, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
        [  # V7: V3 untyped
            ColumnDef("name", RelevanceLevel.PREFERRED, "Entity name", semantic_concept="name"),
            ColumnDef("count", RelevanceLevel.PREFERRED, "Count", is_measurement=True, semantic_concept="count"),
            ColumnDef("rankType", RelevanceLevel.ACCEPTABLE, "Ranking category", semantic_concept="rankType", is_bind_label=True),
        ],
        [  # V8: V4 untyped
            ColumnDef("department", RelevanceLevel.PREFERRED, "Department URI", semantic_concept="department"),
            ColumnDef("facultyCount", RelevanceLevel.PREFERRED, "Faculty count", is_measurement=True, semantic_concept="facultyCount"),
            ColumnDef("stop", RelevanceLevel.PREFERRED, "Stop URI", semantic_concept="stop"),
            ColumnDef("tripCount", RelevanceLevel.PREFERRED, "Trip count", is_measurement=True, semantic_concept="tripCount"),
        ],
    ],
    notes="Cross-dataset: EDU + TRN. 8 variants: normalized/domain-specific × COUNT/COUNT DISTINCT × typed/untyped.",
)

# CROSS04: Hierarchical structures comparison
CROSS04 = AdaptiveGroundTruth(
    query_id="CROSS04",
    query_set="CROSS",
    description="Organizations with their sub-entities",
    dataset="EDU",
    datasets=["EDU", "NRG"],
    query="Show universities with their departments and companies with their operated fields.",
    sparql_queries=["""
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?parent ?parentName ?child ?childName ?hierarchyType WHERE {
    {
        ?child a eduo:Division .
        ?child eduo:partOf ?parent .
        ?parent a eduo:Academy .
        OPTIONAL { ?parent eduo:label ?parentName }
        OPTIONAL { ?child eduo:label ?childName }
        BIND("university_department" AS ?hierarchyType)
    }
    UNION
    {
        ?child a eno:Deposit .
        ?parent a eno:Company .
        ?child eno:activeDepositOperator ?parent .
        OPTIONAL { ?parent eno:designation ?parentName }
        OPTIONAL { ?child eno:designation ?childName }
        BIND("company_field" AS ?hierarchyType)
    }
}
ORDER BY ?hierarchyType ?parentName ?childName
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?universityName ?departmentName ?companyName ?fieldName WHERE {
    {
        ?dept a eduo:Division .
        ?dept eduo:partOf ?uni .
        ?uni a eduo:Academy .
        OPTIONAL { ?uni eduo:label ?universityName }
        OPTIONAL { ?dept eduo:label ?departmentName }
    }
    UNION
    {
        ?field a eno:Deposit .
        ?company a eno:Company .
        ?field eno:activeDepositOperator ?company .
        OPTIONAL { ?company eno:designation ?companyName }
        OPTIONAL { ?field eno:designation ?fieldName }
    }
}
ORDER BY ?universityName ?departmentName ?companyName ?fieldName
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX eno: <http://example.org/ontology/energy#>
SELECT DISTINCT ?parent ?parentName ?child ?childName ?hierarchyType WHERE {
    {
        ?child a eduo:Division .
        ?child eduo:partOf ?parent .
        ?parent a eduo:Academy .
        OPTIONAL { ?parent eduo:label ?parentName }
        OPTIONAL { ?child eduo:label ?childName }
        BIND("university_department" AS ?hierarchyType)
    }
    UNION
    {
        ?child a eno:Deposit .
        ?child eno:activeDepositOperator ?parent .
        OPTIONAL { ?parent eno:designation ?parentName }
        OPTIONAL { ?child eno:designation ?childName }
        BIND("company_field" AS ?hierarchyType)
    }
}
ORDER BY ?hierarchyType ?parentName ?childName
    """.strip(), """
PREFIX eduo: <http://example.org/ontology/education#>
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?universityName ?departmentName ?companyName ?fieldName WHERE {
    {
        ?dept a eduo:Division .
        ?dept eduo:partOf ?uni .
        ?uni a eduo:Academy .
        OPTIONAL { ?uni eduo:label ?universityName }
        OPTIONAL { ?dept eduo:label ?departmentName }
    }
    UNION
    {
        ?field a eno:Deposit .
        ?field eno:activeDepositOperator ?company .
        OPTIONAL { ?company eno:designation ?companyName }
        OPTIONAL { ?field eno:designation ?fieldName }
    }
}
ORDER BY ?universityName ?departmentName ?companyName ?fieldName
    """.strip()],
    columns=[
        [  # Query 1: normalized ?parent/?child + ?hierarchyType
            ColumnDef("parent", RelevanceLevel.ACCEPTABLE, "Parent entity URI", semantic_concept="parent"),
            ColumnDef("parentName", RelevanceLevel.PREFERRED, "Parent name", semantic_concept="parent"),
            ColumnDef("child", RelevanceLevel.ACCEPTABLE, "Child entity URI", semantic_concept="child"),
            ColumnDef("childName", RelevanceLevel.PREFERRED, "Child name", semantic_concept="child"),
            ColumnDef("hierarchyType", RelevanceLevel.ACCEPTABLE, "Hierarchy type", semantic_concept="hierarchyType", is_bind_label=True),
        ],
        [  # Query 2: domain-specific variables
            ColumnDef("universityName", RelevanceLevel.PREFERRED, "University name", semantic_concept="university"),
            ColumnDef("departmentName", RelevanceLevel.PREFERRED, "Department name", semantic_concept="department"),
            ColumnDef("companyName", RelevanceLevel.PREFERRED, "Company name", semantic_concept="company"),
            ColumnDef("fieldName", RelevanceLevel.PREFERRED, "Field name", semantic_concept="field"),
        ],
        [  # Query 3: V1 untyped
            ColumnDef("parent", RelevanceLevel.ACCEPTABLE, "Parent entity URI", semantic_concept="parent"),
            ColumnDef("parentName", RelevanceLevel.PREFERRED, "Parent name", semantic_concept="parent"),
            ColumnDef("child", RelevanceLevel.ACCEPTABLE, "Child entity URI", semantic_concept="child"),
            ColumnDef("childName", RelevanceLevel.PREFERRED, "Child name", semantic_concept="child"),
            ColumnDef("hierarchyType", RelevanceLevel.ACCEPTABLE, "Hierarchy type", semantic_concept="hierarchyType", is_bind_label=True),
        ],
        [  # Query 4: V2 untyped
            ColumnDef("universityName", RelevanceLevel.PREFERRED, "University name", semantic_concept="university"),
            ColumnDef("departmentName", RelevanceLevel.PREFERRED, "Department name", semantic_concept="department"),
            ColumnDef("companyName", RelevanceLevel.PREFERRED, "Company name", semantic_concept="company"),
            ColumnDef("fieldName", RelevanceLevel.PREFERRED, "Field name", semantic_concept="field"),
        ],
    ],
    notes="Cross-dataset: EDU + NRG. Four variants: normalized/domain-specific × typed/untyped ?company.",
)

# CROSS05: Filtered entities with numeric/time conditions
CROSS05 = AdaptiveGroundTruth(
    query_id="CROSS05",
    query_set="CROSS",
    description="Deep wellbores and early morning trips",
    dataset="NRG",
    datasets=["NRG", "TRN"],
    query="Find wellbores deeper than 3000 meters and trips departing before 07:00.",
    sparql_queries=["""
PREFIX eno: <http://example.org/ontology/energy#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT DISTINCT ?entity ?name ?value ?filterType WHERE {
    {
        ?entity a eno:Borehole .
        ?entity eno:designation ?name .
        ?entity eno:totalDrillDepth ?value .
        FILTER(?value > 3000)
        BIND("deep_wellbore" AS ?filterType)
    }
    UNION
    {
        ?stopTime a tro:StopEvent .
        ?entity a tro:Trip .
        ?stopTime tro:journey ?entity .
        ?stopTime tro:leaveTime ?value .
        FILTER(?value < "07:00:00")
        OPTIONAL { ?entity tro:abbreviation ?name }
        BIND("early_trip" AS ?filterType)
    }
}
ORDER BY ?filterType ?value
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?wellbore ?wellboreName ?depth ?trip ?departureTime WHERE {
    {
        ?wellbore a eno:Borehole .
        ?wellbore eno:totalDrillDepth ?depth .
        FILTER(?depth > 3000)
        OPTIONAL { ?wellbore eno:designation ?wellboreName }
    }
    UNION
    {
        ?stopTime a tro:StopEvent .
        ?trip a tro:Trip .
        ?stopTime tro:journey ?trip .
        ?stopTime tro:leaveTime ?departureTime .
        FILTER(?departureTime < "07:00:00")
    }
}
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?wellbore ?wellboreName ?depth ?trip ?departureTime WHERE {
    {
        ?wellbore a eno:Borehole .
        ?wellbore eno:finalVerticalDrillDepth ?depth .
        FILTER(?depth > 3000)
        OPTIONAL { ?wellbore eno:designation ?wellboreName }
    }
    UNION
    {
        ?stopTime a tro:StopEvent .
        ?trip a tro:Trip .
        ?stopTime tro:journey ?trip .
        ?stopTime tro:leaveTime ?departureTime .
        FILTER(?departureTime < "07:00:00")
    }
}
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT DISTINCT ?entity ?name ?value ?filterType WHERE {
    {
        ?entity a eno:Borehole .
        ?entity eno:designation ?name .
        ?entity eno:totalDrillDepth ?value .
        FILTER(?value > 3000)
        BIND("deep_wellbore" AS ?filterType)
    }
    UNION
    {
        ?stopTime a tro:StopEvent .
        ?stopTime tro:journey ?entity .
        ?stopTime tro:leaveTime ?value .
        FILTER(?value < "07:00:00")
        OPTIONAL { ?entity tro:abbreviation ?name }
        BIND("early_trip" AS ?filterType)
    }
}
ORDER BY ?filterType ?value
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?wellbore ?wellboreName ?depth ?trip ?departureTime WHERE {
    {
        ?wellbore a eno:Borehole .
        ?wellbore eno:totalDrillDepth ?depth .
        FILTER(?depth > 3000)
        OPTIONAL { ?wellbore eno:designation ?wellboreName }
    }
    UNION
    {
        ?stopTime a tro:StopEvent .
        ?stopTime tro:journey ?trip .
        ?stopTime tro:leaveTime ?departureTime .
        FILTER(?departureTime < "07:00:00")
    }
}
    """.strip(), """
PREFIX eno: <http://example.org/ontology/energy#>
PREFIX tro: <http://example.org/ontology/transport#>
SELECT ?wellbore ?wellboreName ?depth ?trip ?departureTime WHERE {
    {
        ?wellbore a eno:Borehole .
        ?wellbore eno:finalVerticalDrillDepth ?depth .
        FILTER(?depth > 3000)
        OPTIONAL { ?wellbore eno:designation ?wellboreName }
    }
    UNION
    {
        ?stopTime a tro:StopEvent .
        ?stopTime tro:journey ?trip .
        ?stopTime tro:leaveTime ?departureTime .
        FILTER(?departureTime < "07:00:00")
    }
}
    """.strip()],
    columns=[
        [  # V1: normalized ?entity/?name/?value + ?filterType
            ColumnDef("entity", RelevanceLevel.ACCEPTABLE, "Entity URI", semantic_concept="entity"),
            ColumnDef("name", RelevanceLevel.PREFERRED, "Entity name", semantic_concept="name"),
            ColumnDef("value", RelevanceLevel.PREFERRED, "Filter value (depth/time)", semantic_concept="value"),
            ColumnDef("filterType", RelevanceLevel.ACCEPTABLE, "Filter category", semantic_concept="filterType", is_bind_label=True),
        ],
        [  # V2: domain-specific with totalDepth
            ColumnDef("wellbore", RelevanceLevel.PREFERRED, "Wellbore URI", semantic_concept="wellbore"),
            ColumnDef("wellboreName", RelevanceLevel.PREFERRED, "Wellbore name", semantic_concept="wellbore"),
            ColumnDef("depth", RelevanceLevel.PREFERRED, "Depth value", semantic_concept="depth"),
            ColumnDef("trip", RelevanceLevel.PREFERRED, "Trip URI", semantic_concept="trip"),
            ColumnDef("departureTime", RelevanceLevel.PREFERRED, "Departure time", semantic_concept="departureTime"),
        ],
        [  # V3: domain-specific with verticalDepth
            ColumnDef("wellbore", RelevanceLevel.PREFERRED, "Wellbore URI", semantic_concept="wellbore"),
            ColumnDef("wellboreName", RelevanceLevel.PREFERRED, "Wellbore name", semantic_concept="wellbore"),
            ColumnDef("depth", RelevanceLevel.PREFERRED, "Depth value", semantic_concept="depth"),
            ColumnDef("trip", RelevanceLevel.PREFERRED, "Trip URI", semantic_concept="trip"),
            ColumnDef("departureTime", RelevanceLevel.PREFERRED, "Departure time", semantic_concept="departureTime"),
        ],
        [  # V4: V1 untyped (no ?entity a tro:Trip)
            ColumnDef("entity", RelevanceLevel.ACCEPTABLE, "Entity URI", semantic_concept="entity"),
            ColumnDef("name", RelevanceLevel.PREFERRED, "Entity name", semantic_concept="name"),
            ColumnDef("value", RelevanceLevel.PREFERRED, "Filter value (depth/time)", semantic_concept="value"),
            ColumnDef("filterType", RelevanceLevel.ACCEPTABLE, "Filter category", semantic_concept="filterType", is_bind_label=True),
        ],
        [  # V5: V2 untyped
            ColumnDef("wellbore", RelevanceLevel.PREFERRED, "Wellbore URI", semantic_concept="wellbore"),
            ColumnDef("wellboreName", RelevanceLevel.PREFERRED, "Wellbore name", semantic_concept="wellbore"),
            ColumnDef("depth", RelevanceLevel.PREFERRED, "Depth value", semantic_concept="depth"),
            ColumnDef("trip", RelevanceLevel.PREFERRED, "Trip URI", semantic_concept="trip"),
            ColumnDef("departureTime", RelevanceLevel.PREFERRED, "Departure time", semantic_concept="departureTime"),
        ],
        [  # V6: V3 untyped
            ColumnDef("wellbore", RelevanceLevel.PREFERRED, "Wellbore URI", semantic_concept="wellbore"),
            ColumnDef("wellboreName", RelevanceLevel.PREFERRED, "Wellbore name", semantic_concept="wellbore"),
            ColumnDef("depth", RelevanceLevel.PREFERRED, "Depth value", semantic_concept="depth"),
            ColumnDef("trip", RelevanceLevel.PREFERRED, "Trip URI", semantic_concept="trip"),
            ColumnDef("departureTime", RelevanceLevel.PREFERRED, "Departure time", semantic_concept="departureTime"),
        ],
    ],
    notes="Cross-dataset: NRG + TRN. Six variants: normalized/totalDepth/verticalDepth × typed/untyped ?trip.",
)

# Collect all CROSS queries
SET_CROSS = [CROSS01, CROSS02, CROSS03, CROSS04, CROSS05]


# =============================================================================
# COLLECTION AND HELPER FUNCTIONS
# =============================================================================

# All queries by set
ALL_EXPERIMENTAL_QUERIES = (
    SET_BASE + SET_SYN + SET_TYPO + SET_UNDER + SET_CROSS
)

# Paired queries (BASE with their SYN and TYPO variants)
PAIRED_QUERIES: dict[str, dict[str, AdaptiveGroundTruth]] = {}
for base in SET_BASE:
    base_num = base.query_id[4:]  # "BASE01" -> "01"
    PAIRED_QUERIES[base.query_id] = {
        "baseline": base,
        "synonym": next((s for s in SET_SYN if s.query_id == f"SYN{base_num}"), None),
        "typo": next((t for t in SET_TYPO if t.query_id == f"TYPO{base_num}"), None),
    }


def get_experimental_query(query_id: str) -> AdaptiveGroundTruth | None:
    """Get an experimental query by ID."""
    for query in ALL_EXPERIMENTAL_QUERIES:
        if query.query_id == query_id:
            return query
    return None


def get_queries_by_set(query_set: str) -> list[AdaptiveGroundTruth]:
    """Get all queries for a specific set."""
    return [q for q in ALL_EXPERIMENTAL_QUERIES if q.query_set == query_set]


def get_paired_queries(base_id: str) -> dict[str, AdaptiveGroundTruth | None]:
    """Get baseline and its variants for paired comparison."""
    return PAIRED_QUERIES.get(base_id, {})


def get_baseline_for_variant(variant_id: str) -> AdaptiveGroundTruth | None:
    """Get the baseline query for a SYN or TYPO variant."""
    if variant_id.startswith("SYN"):
        base_num = variant_id[3:]
    elif variant_id.startswith("TYPO"):
        base_num = variant_id[4:]
    else:
        return None
    base_id = f"BASE{base_num}"
    return get_experimental_query(base_id)


# Statistics
CORPUS_STATS = {
    "total": len(ALL_EXPERIMENTAL_QUERIES),
    "baseline": len(SET_BASE),
    "synonym": len(SET_SYN),
    "typo": len(SET_TYPO),
    "underspecified": len(SET_UNDER),
    "cross_dataset": len(SET_CROSS),
    "paired_total": len(SET_BASE) * 3,  # BASE + SYN + TYPO
}


__all__ = [
    # Data structures
    "AmbiguityLayer",
    "ExperimentalQuery",
    # Query sets
    "SET_BASE",
    "SET_SYN",
    "SET_TYPO",
    "SET_UNDER",
    "SET_CROSS",
    "ALL_EXPERIMENTAL_QUERIES",
    "PAIRED_QUERIES",
    # Helper functions
    "get_experimental_query",
    "get_queries_by_set",
    "get_paired_queries",
    "get_baseline_for_variant",
    # Statistics
    "CORPUS_STATS",
]