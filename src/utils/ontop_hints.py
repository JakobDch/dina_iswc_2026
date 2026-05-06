"""
OnTop-specific performance hints for SPARQL query optimization.

These hints help the SPARQL agent optimize queries that cause timeouts
in OnTop (OBDA middleware) due to inefficient SPARQL-to-SQL translation.

Sources:
- https://github.com/ontop/ontop/issues/485 (Shared property URIs → UNION ALL)
- https://github.com/ontop/ontop/issues/800 (N-columns → N-1 Self-Joins)
"""

ONTOP_TIMEOUT_HINTS = """
## OnTop Performance Optimization Guide

Your query timed out. In OnTop (OBDA), SPARQL is translated to SQL.
Certain patterns cause expensive SQL operations:

### 1. CARTESIAN PRODUCTS (Most Common!)
**Problem:** Disconnected triple patterns create CROSS JOINs.
```sparql
# BAD - no connection between ?a and ?b
?a a :TypeA . ?b a :TypeB .

# GOOD - connected via property
?a a :TypeA . ?a :relatedTo ?b . ?b a :TypeB .
```
**Fix:** Ensure ALL triple patterns share variables.

### 2. TOO MANY SELECT VARIABLES (Self-Joins)
**Problem:** Each non-key column causes a self-join in SQL.
Source: https://github.com/ontop/ontop/issues/800
```sparql
# SLOW - 10 variables = 9 self-joins
SELECT ?a ?b ?c ?d ?e ?f ?g ?h ?i ?j WHERE {...}

# FASTER - fewer variables
SELECT ?a ?b ?c WHERE {...}
```
**Fix:** Request only essential variables. Split into multiple queries if needed.

### 3. MULTIPLE OPTIONAL CLAUSES (Left Joins)
**Problem:** Each OPTIONAL becomes a LEFT JOIN in SQL.
```sparql
# SLOW - 3 left joins
OPTIONAL { ?x :prop1 ?v1 }
OPTIONAL { ?x :prop2 ?v2 }
OPTIONAL { ?x :prop3 ?v3 }

# FASTER - single optional block
OPTIONAL { ?x :prop1 ?v1 ; :prop2 ?v2 ; :prop3 ?v3 }
```

### 4. UNBOUNDED PATTERNS
**Problem:** Generic patterns scan entire database.
```sparql
# BAD - scans everything
?s ?p ?o
?x a ?type

# GOOD - specific classes/properties
?x a :SpecificClass . ?x :specificProp ?value .
```

### 5. REGEX IN FILTER
**Problem:** REGEX cannot use database indexes.
```sparql
# SLOW - full table scan
FILTER(REGEX(?name, "pattern"))

# FASTER - can use indexes
FILTER(CONTAINS(?name, "pattern"))
FILTER(STRSTARTS(?name, "prefix"))
```

### 6. MISSING LIMIT
Always add LIMIT for exploratory queries.

### Recommendation
1. Simplify: Remove unnecessary variables/OPTIONAL
2. Connect: Ensure all patterns share variables
3. Limit: Add LIMIT clause
4. Test incrementally: Start with basic pattern, add constraints
"""


def get_ontop_timeout_hints() -> str:
    """Return detailed OnTop-specific hints for query timeout optimization.

    These hints are shown to the SPARQL agent when a query times out,
    helping it understand common OnTop performance issues and how to fix them.

    Returns:
        A formatted string with optimization recommendations.
    """
    return ONTOP_TIMEOUT_HINTS
