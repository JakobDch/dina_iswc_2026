"""
OnTop Query Validator - Pre-execution validation to catch known problematic patterns.

This module validates SPARQL queries BEFORE sending them to OnTop, catching patterns
that are known to fail and providing actionable feedback to the agent.

Patterns are GENERAL OnTop limitations, not dataset-specific.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Query will definitely fail - don't execute
    WARNING = "warning"  # Query might fail or be very slow - execute with caution
    INFO = "info"        # Potential issue but query might work


@dataclass
class ValidationResult:
    """Result of query validation."""
    is_valid: bool
    severity: Optional[ValidationSeverity]
    error_type: Optional[str]
    message: Optional[str]
    suggestion: Optional[str]
    matched_pattern: Optional[str] = None


# =============================================================================
# ANTI-PATTERNS - General OnTop limitations (NOT dataset-specific!)
# =============================================================================

ONTOP_ANTI_PATTERNS = [
    # =========================================================================
    # ERROR PATTERNS - Query will definitely fail, don't execute
    # =========================================================================

    # -------------------------------------------------------------------------
    # ERROR: Type casts in aggregation functions
    # Response: SQLSerializationException: Only DBFunctionSymbols must be provided
    # General OnTop limitation: Cannot translate XSD casts inside aggregations to SQL
    # -------------------------------------------------------------------------
    {
        "name": "type_cast_in_avg",
        "pattern": r"AVG\s*\(\s*xsd:(decimal|integer|float|double)\s*\(",
        "severity": ValidationSeverity.ERROR,
        "error_type": "SQLSerializationException",
        "message": "OnTop cannot translate type casts (xsd:decimal, xsd:integer, etc.) inside AVG() to SQL.",
        "suggestion": "Use AVG(?variable) directly without type cast. The datatype must already be numeric in the database.",
        "examples": {
            "wrong": "AVG(xsd:decimal(?rating))",
            "correct": "AVG(?rating)"
        }
    },
    {
        "name": "type_cast_in_sum",
        "pattern": r"SUM\s*\(\s*xsd:(decimal|integer|float|double)\s*\(",
        "severity": ValidationSeverity.ERROR,
        "error_type": "SQLSerializationException",
        "message": "OnTop cannot translate type casts inside SUM() to SQL.",
        "suggestion": "Use SUM(?variable) directly without type cast.",
        "examples": {
            "wrong": "SUM(xsd:integer(?count))",
            "correct": "SUM(?count)"
        }
    },
    {
        "name": "arithmetic_in_avg",
        "pattern": r"AVG\s*\(\s*\([^)]*xsd:(decimal|integer)[^)]*\)",
        "severity": ValidationSeverity.ERROR,
        "error_type": "SQLSerializationException",
        "message": "OnTop cannot translate arithmetic expressions with type casts inside AVG() to SQL.",
        "suggestion": "Compute averages directly on numeric columns, not on expressions with casts.",
        "examples": {
            "wrong": "AVG((xsd:integer(?r1) + xsd:integer(?r2)) / 2)",
            "correct": "AVG(?rating)"
        }
    },

    # -------------------------------------------------------------------------
    # ERROR: Property paths (*, +, ?)
    # Response: OntopUnsupportedInputQueryException: Unsupported arbitrary length path
    # General OnTop limitation: Property paths are not implemented
    # Patterns match: prefix:*, prefix:property*, prefix:prop/prefix:prop*
    # -------------------------------------------------------------------------
    {
        "name": "star_property_path",
        # Matches: ?s prefix:* ?o, ?s prefix:prop* ?o, ?s prefix:prop/prefix:prop* ?o
        "pattern": r"[?$]\w+\s+[\w:/]+\*\s+[?$]\w+",
        "severity": ValidationSeverity.ERROR,
        "error_type": "ArbitraryLengthPath",
        "message": "OnTop does not support '*' property paths (zero or more).",
        "suggestion": "Use explicit triple patterns or UNION for fixed path lengths.",
        "examples": {
            "wrong": "?x rdfs:subClassOf* ?y",
            "correct": "{ ?x rdfs:subClassOf ?y } UNION { BIND(?x AS ?y) }"
        }
    },
    {
        "name": "plus_property_path",
        # Matches: ?s prefix:prop+ ?o, ?s prefix:prop/prefix:prop+ ?o
        "pattern": r"[?$]\w+\s+[\w:/]+\+\s+[?$]\w+",
        "severity": ValidationSeverity.ERROR,
        "error_type": "ArbitraryLengthPath",
        "message": "OnTop does not support '+' property paths (one or more).",
        "suggestion": "Write paths as separate triple patterns.",
        "examples": {
            "wrong": "?x rdfs:subClassOf+ ?y",
            "correct": "?x rdfs:subClassOf ?mid . ?mid rdfs:subClassOf ?y"
        }
    },
    {
        "name": "optional_property_path",
        # Matches: ?s prefix:prop? ?o (zero or one)
        "pattern": r"[?$]\w+\s+[\w:/]+\?\s+[?$]\w+",
        "severity": ValidationSeverity.ERROR,
        "error_type": "ArbitraryLengthPath",
        "message": "OnTop does not support '?' property paths (zero or one).",
        "suggestion": "Use OPTIONAL for optional paths.",
        "examples": {
            "wrong": "?x foaf:knows? ?y",
            "correct": "OPTIONAL { ?x foaf:knows ?y }"
        }
    },

    # -------------------------------------------------------------------------
    # ERROR: Nested aggregations
    # Response: InvalidSPARQL - not allowed in SPARQL standard
    # -------------------------------------------------------------------------
    {
        "name": "nested_aggregation",
        "pattern": r"(AVG|SUM|COUNT)\s*\(\s*(AVG|SUM|COUNT)\s*\(",
        "severity": ValidationSeverity.ERROR,
        "error_type": "InvalidSPARQL",
        "message": "Nested aggregation functions are not allowed in SPARQL.",
        "suggestion": "Use subqueries to nest aggregations.",
        "examples": {
            "wrong": "AVG(COUNT(?x))",
            "correct": "SELECT (AVG(?cnt) AS ?avg) WHERE { SELECT (COUNT(?x) AS ?cnt) WHERE {...} GROUP BY ?y }"
        }
    },

    # =========================================================================
    # WARNING PATTERNS - Query might fail or be slow, execute with caution
    # =========================================================================

    # -------------------------------------------------------------------------
    # WARNING: AVG on pre-aggregated variables
    # Response: SQLSerializationException
    # General issue: AVG on computed/aggregated values often fails
    # -------------------------------------------------------------------------
    {
        "name": "avg_with_subquery_variable",
        "pattern": r"AVG\s*\(\s*\?[a-zA-Z]*[Aa]vg[a-zA-Z]*\s*\)",
        "severity": ValidationSeverity.WARNING,
        "error_type": "SQLSerializationException",
        "message": "AVG() on a variable from a subquery or with 'avg' in its name may fail.",
        "suggestion": "Use AVG() directly on the original property, not on computed/aggregated values.",
        "examples": {
            "wrong": "AVG(?avgReviewRating)",
            "correct": "AVG(?rating)"
        }
    },

    # -------------------------------------------------------------------------
    # WARNING: GROUP_CONCAT with type casts
    # Response: SQLSerializationException
    # -------------------------------------------------------------------------
    {
        "name": "group_concat_complex",
        "pattern": r"GROUP_CONCAT\s*\([^)]*xsd:(decimal|integer)",
        "severity": ValidationSeverity.WARNING,
        "error_type": "SQLSerializationException",
        "message": "GROUP_CONCAT with type casts may cause translation issues.",
        "suggestion": "Use GROUP_CONCAT only with simple variables.",
        "examples": {
            "wrong": "GROUP_CONCAT(xsd:string(?val))",
            "correct": "GROUP_CONCAT(?val; SEPARATOR=', ')"
        }
    },

    # -------------------------------------------------------------------------
    # WARNING: Multiple GROUP_CONCAT in one query
    # Response: Timeout or serialization issues
    # General issue: Multiple string aggregations are expensive
    # -------------------------------------------------------------------------
    {
        "name": "multiple_group_concat",
        "pattern": r"GROUP_CONCAT.*GROUP_CONCAT.*GROUP_CONCAT",
        "severity": ValidationSeverity.WARNING,
        "error_type": "QueryTimeout",
        "message": "Multiple GROUP_CONCAT aggregations in one query are expensive.",
        "suggestion": "Split into separate queries or remove GROUP_CONCATs and fetch data individually.",
        "examples": {
            "wrong": "SELECT (GROUP_CONCAT(?a) AS ?list1) (GROUP_CONCAT(?b) AS ?list2) ...",
            "correct": "Run separate queries for each aggregation"
        }
    },

    # -------------------------------------------------------------------------
    # WARNING: Multiple OPTIONALs
    # Response: SQLSerializationException or timeout
    # General OnTop limitation: Many OPTIONALs create complex LEFT JOINs
    # -------------------------------------------------------------------------
    {
        "name": "multi_optional",
        "pattern": r"OPTIONAL\s*\{[^}]*\}\s*OPTIONAL\s*\{[^}]*\}\s*OPTIONAL",
        "severity": ValidationSeverity.WARNING,
        "error_type": "SQLSerializationException",
        "message": "Multiple OPTIONAL blocks often cause OnTop translation errors or slow queries.",
        "suggestion": "Reduce the number of OPTIONALs or split into separate queries.",
        "examples": {
            "wrong": "OPTIONAL { ... } OPTIONAL { ... } OPTIONAL { ... }",
            "correct": "Run separate queries for each OPTIONAL part"
        }
    },

    # -------------------------------------------------------------------------
    # WARNING: VALUES clause
    # Response: NullPointerException in certain combinations
    # General OnTop limitation: VALUES handling can be buggy
    # -------------------------------------------------------------------------
    {
        "name": "values_clause_warning",
        "pattern": r"\bVALUES\s+\?\w+\s*\{",
        "severity": ValidationSeverity.INFO,
        "error_type": "NullPointerException",
        "message": "VALUES clauses can cause OnTop internal errors in certain combinations.",
        "suggestion": "If the query fails, replace VALUES with FILTER using IN or UNION.",
        "examples": {
            "wrong": 'VALUES ?name { "Value1" "Value2" }',
            "correct": 'FILTER(?name IN ("Value1", "Value2"))'
        }
    },

    # -------------------------------------------------------------------------
    # WARNING: Superclass queries that trigger UNION mapping issues
    # Response: Multiple entries with same key: UNION
    # General issue: When mappings use UNION for subclasses, querying superclass fails
    # -------------------------------------------------------------------------
    {
        "name": "superclass_union_mapping",
        "pattern": r"\?\w+\s+a\s+\w+:(?:Student|Person|Entity|Thing)\s*\.\s*\?\w+\s+\w+:\w+\s+\?\w+",
        "severity": ValidationSeverity.INFO,
        "error_type": "UnionMappingBug",
        "message": "Querying a superclass that has UNION mappings for subclasses may cause internal errors.",
        "suggestion": "Try querying specific subclasses instead of the superclass.",
        "examples": {
            "wrong": "?s a :Student . ?s :memberOf ?dept",
            "correct": "?s a :DoctoralCandidate . ?s :memberOf ?dept"
        }
    },

    # -------------------------------------------------------------------------
    # WARNING: GROUP BY on large join results without LIMIT
    # Response: Timeout or ColumnIDError
    # General issue: Aggregating over unbounded joins is expensive
    # -------------------------------------------------------------------------
    {
        "name": "group_by_without_limit",
        "pattern": r"GROUP\s+BY\s+\?\w+(?!\s*\n*.*LIMIT)",
        "severity": ValidationSeverity.INFO,
        "error_type": "QueryTimeout",
        "message": "GROUP BY without LIMIT can cause timeouts on large result sets.",
        "suggestion": "Add LIMIT to the query or use a subquery with LIMIT before grouping.",
        "examples": {
            "wrong": "SELECT ?x (COUNT(?y) AS ?cnt) WHERE { ... } GROUP BY ?x",
            "correct": "SELECT ?x (COUNT(?y) AS ?cnt) WHERE { ... } GROUP BY ?x LIMIT 100"
        }
    },

    # -------------------------------------------------------------------------
    # WARNING: DATATYPE() on object properties with large tables
    # Response: Timeout
    # General issue: DATATYPE on URIs with large scans is slow
    # -------------------------------------------------------------------------
    {
        "name": "datatype_large_scan",
        "pattern": r"DATATYPE\s*\(\s*\?\w+\s*\).*\?\w+\s+\?\w+\s+\?\w+",
        "severity": ValidationSeverity.INFO,
        "error_type": "QueryTimeout",
        "message": "DATATYPE() combined with unbounded patterns may cause timeouts.",
        "suggestion": "Remove DATATYPE() if you only need values, or test with a simple SELECT first.",
        "examples": {
            "wrong": "SELECT ?value (DATATYPE(?value) AS ?dt) WHERE { ?s ?p ?value }",
            "correct": "SELECT ?value WHERE { ?s specificProp ?value } LIMIT 10"
        }
    },
]


def validate_query(query: str) -> ValidationResult:
    """
    Validate a SPARQL query against known OnTop anti-patterns.

    Args:
        query: The SPARQL query string to validate

    Returns:
        ValidationResult with validation status and feedback
    """
    if not query or not query.strip():
        return ValidationResult(
            is_valid=False,
            severity=ValidationSeverity.ERROR,
            error_type="EmptyQuery",
            message="Query ist leer.",
            suggestion="Generiere eine gültige SPARQL Query."
        )

    # Check each anti-pattern
    for pattern_def in ONTOP_ANTI_PATTERNS:
        regex = pattern_def["pattern"]
        flags = re.IGNORECASE | re.DOTALL

        if re.search(regex, query, flags):
            return ValidationResult(
                is_valid=pattern_def["severity"] != ValidationSeverity.ERROR,
                severity=pattern_def["severity"],
                error_type=pattern_def["error_type"],
                message=pattern_def["message"],
                suggestion=pattern_def["suggestion"],
                matched_pattern=pattern_def["name"]
            )

    # No issues found
    return ValidationResult(
        is_valid=True,
        severity=None,
        error_type=None,
        message=None,
        suggestion=None
    )


def validate_query_with_feedback(query: str) -> tuple[bool, str]:
    """
    Validate a query and return a tuple of (should_execute, feedback_message).

    This is the main function to use in the orchestrator before query execution.

    Args:
        query: The SPARQL query to validate

    Returns:
        Tuple of (should_execute: bool, feedback: str or None)
        - should_execute: True if query should be sent to OnTop
        - feedback: Actionable feedback message if validation failed, None otherwise
    """
    result = validate_query(query)

    if result.severity == ValidationSeverity.ERROR:
        feedback = f"""
VALIDATION ERROR: {result.message}

{result.suggestion}

Bitte generiere eine korrigierte Query.
"""
        return False, feedback.strip()

    elif result.severity == ValidationSeverity.WARNING:
        feedback = f"""
VALIDATION WARNING: {result.message}

{result.suggestion}

Die Query wird trotzdem ausgeführt, aber könnte fehlschlagen.
"""
        return True, feedback.strip()

    return True, None


def get_feedback_for_ontop_error(error_message: str, original_query: str) -> str:
    """
    Generate specific feedback based on OnTop error message and the original query.

    This is used AFTER a query fails to provide actionable feedback to the agent.

    Args:
        error_message: The error message from OnTop
        original_query: The query that failed

    Returns:
        Specific, actionable feedback for the agent
    """
    feedback_parts = []

    # SQLSerializationException - Type cast issues
    if "SQLSerializationException" in error_message:
        if "Only DBFunctionSymbols" in error_message:
            feedback_parts.append("""
ERROR: OnTop kann diesen SPARQL-Ausdruck nicht in SQL übersetzen.

HÄUFIGE URSACHEN:
1. Type-Casts in Aggregationsfunktionen: AVG(xsd:decimal(?x)) ist NICHT unterstützt
2. Komplexe Ausdrücke in Aggregationen: AVG((?a + ?b) / 2) ist NICHT unterstützt

LÖSUNG:
- Verwende AVG(?variable) direkt ohne Konvertierung
- Die Spalte muss bereits numerisch in der Datenbank sein
- Falls Berechnung nötig: Berechne zuerst in einer Subquery, dann aggregiere
""")
        elif "does not appear in columnIDs" in error_message:
            feedback_parts.append("""
ERROR: Eine Variable in der Query ist nicht korrekt definiert.

LÖSUNG:
- Prüfe ob alle SELECTierten Variablen im WHERE-Teil definiert sind
- Bei Subqueries: Stelle sicher dass Variablen korrekt durchgereicht werden
""")

    # Query timeout
    elif "maximum statement execution time exceeded" in error_message:
        feedback_parts.append("""
ERROR: Query-Timeout - die Ausführung dauerte zu lange.

URSACHEN bei diesem Dataset:
1. Zu viele Ergebnisse ohne LIMIT
2. Teure JOINs über große Tabellen (besonders EDU Publication/Author)
3. DISTINCT oder GROUP BY über große Ergebnismengen

LÖSUNGEN:
1. Füge LIMIT hinzu (z.B. LIMIT 100)
2. Verwende eine Subquery um erst zu filtern: { SELECT ?x WHERE {...} LIMIT 100 }
3. Füge spezifische FILTER hinzu um die Datenmenge einzuschränken
4. Bei EDU: Vermeide Queries auf Publication ohne Filter
""")

    # UNION mapping bug
    elif "Multiple entries with same key: UNION" in error_message:
        feedback_parts.append("""
ERROR: OnTop UNION-Mapping Bug

URSACHE:
Das Mapping verwendet UNION für Subklassen und diese Query triggert einen internen Fehler.

LÖSUNG:
- Verwende spezifische Subklassen statt der Superklasse
- Beispiel: Statt "?s a eduo:Student" verwende "?s a eduo:DoctoralCandidate"
- Oder: Umgehe das Property das den Fehler verursacht
""")

    # Property paths
    elif "ArbitraryLengthPath" in error_message or "Unsupported arbitrary length path" in error_message:
        feedback_parts.append("""
ERROR: Property Paths (*, +, ?) werden von OnTop NICHT unterstützt!

LÖSUNG:
- Ersetze Wildcards durch explizite Properties
- Statt "?s prefix:* ?o" → "?s prefix:specificProperty ?o"
- Für transitive Pfade: Schreibe mehrere Triple Patterns oder verwende UNION
""")

    # NullPointerException
    elif "NullPointerException" in error_message:
        feedback_parts.append("""
ERROR: OnTop interner Fehler (NullPointerException)

URSACHE:
Die Query-Struktur verursacht einen internen OnTop-Fehler, oft bei:
- Komplexen VALUES Klauseln
- Bestimmten OPTIONAL Kombinationen
- Ungewöhnlichen Filter-Patterns

LÖSUNG:
- Vereinfache die Query
- Entferne VALUES Klauseln und verwende stattdessen FILTER
- Teste Teile der Query einzeln
""")

    # Communications link failure
    elif "CommunicationsException" in error_message or "Communications link failure" in error_message:
        feedback_parts.append("""
ERROR: Verbindung zum Datenbank-Server verloren

URSACHE:
Die Query war so teuer dass die Datenbankverbindung abgebrochen wurde.

LÖSUNG:
- Drastisch vereinfachen
- LIMIT 10 hinzufügen
- Weniger JOINs verwenden
""")

    # Generic fallback
    if not feedback_parts:
        feedback_parts.append(f"""
ERROR: OnTop Query fehlgeschlagen

Fehler: {error_message[:500]}

ALLGEMEINE TIPPS:
1. Vereinfache die Query
2. Füge LIMIT hinzu
3. Prüfe ob alle Prefixes korrekt definiert sind
4. Teste Teile der Query einzeln
""")

    return "\n".join(feedback_parts).strip()


# =============================================================================
# Test function
# =============================================================================

def run_validation_tests():
    """Run validation tests against known problematic queries from deepseek_test_5."""

    test_queries = [
        # SQLSerializationException - Type casts in AVG
        ("""
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
SELECT ?product (AVG(xsd:decimal(?rating)) AS ?avgRating)
WHERE { ?product bsbm:rating ?rating }
GROUP BY ?product
""", "type_cast_in_avg", ValidationSeverity.ERROR),

        # Arithmetic in AVG
        ("""
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
SELECT ?product (AVG((xsd:integer(?r1) + xsd:integer(?r2)) / 4.0) AS ?avgRating)
WHERE { ?product bsbm:rating1 ?r1 . ?product bsbm:rating2 ?r2 }
GROUP BY ?product
""", "arithmetic_in_avg", ValidationSeverity.ERROR),

        # Wildcard property path
        ("""
PREFIX eno: <http://example.org/ontology/energy#>
SELECT ?value WHERE {
    ?subject a eno:Field .
    ?subject eno:* ?value .
}
""", "wildcard_property_path", ValidationSeverity.ERROR),

        # EDU Publication/Author scan (warning)
        ("""
PREFIX eduo: <http://example.org/ontology/education#>
SELECT ?pub ?author WHERE {
  ?pub a eduo:Publication .
  ?pub eduo:author ?author .
}
LIMIT 5
""", "edu_publication_author_scan", ValidationSeverity.WARNING),

        # EDU Student memberOf (warning)
        ("""
PREFIX eduo: <http://example.org/ontology/education#>
SELECT ?s ?dept WHERE {
    ?s a eduo:Student .
    ?s eduo:memberOf ?dept .
}
""", "edu_student_memberof", ValidationSeverity.WARNING),

        # Valid query - should pass
        ("""
PREFIX bsbm: <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/>
SELECT ?product (AVG(?rating) AS ?avgRating)
WHERE { ?product bsbm:rating ?rating }
GROUP BY ?product
LIMIT 100
""", None, None),
    ]

    print("=" * 70)
    print("OnTop Query Validator - Test Results")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, (query, expected_pattern, expected_severity) in enumerate(test_queries, 1):
        result = validate_query(query)

        # Check if result matches expectation
        pattern_match = result.matched_pattern == expected_pattern
        severity_match = result.severity == expected_severity

        if pattern_match and severity_match:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"\nTest {i}: {status}")
        print(f"  Expected: pattern={expected_pattern}, severity={expected_severity}")
        print(f"  Got:      pattern={result.matched_pattern}, severity={result.severity}")
        if result.message:
            print(f"  Message:  {result.message[:80]}...")

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    run_validation_tests()
