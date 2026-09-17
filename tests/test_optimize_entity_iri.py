"""
Test script for the generic R2RML IRI optimization tools.

These tools work for ANY R2RML mapping file using rdflib to parse
the mapping as an RDF graph.
"""

import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.mapping_optimizer_tools import (
    R2RMLParser,
    analyze_mapping_entities,
    analyze_iri_dependencies,
    _get_mapping_path,
)


def test_r2rml_parser_edu():
    """Test the generic R2RML parser on EDU."""
    print("=" * 60)
    print("TEST 1: R2RML Parser (EDU)")
    print("=" * 60)

    mapping_path = _get_mapping_path("edu")
    if not mapping_path:
        print("ERROR: EDU mapping not found")
        return False

    parser = R2RMLParser(mapping_path)

    print(f"\nFound {len(parser.triples_maps)} TriplesMaps")

    # Test entity analysis
    entities = parser.get_entity_info()
    print(f"Found {len(entities)} distinct entity types")

    print("\nTop 3 entities by usage:")
    for i, e in enumerate(entities[:3]):
        print(f"  {i+1}. Variables: {e['variables']}")
        print(f"     Subjects: {e['subject_count']}, Objects: {e['object_count']}")

    return True


def test_r2rml_parser_trn():
    """Test the generic R2RML parser on TRN."""
    print("\n" + "=" * 60)
    print("TEST 2: R2RML Parser (TRN)")
    print("=" * 60)

    mapping_path = _get_mapping_path("trn")
    if not mapping_path:
        print("ERROR: TRN mapping not found")
        return False

    parser = R2RMLParser(mapping_path)

    print(f"\nFound {len(parser.triples_maps)} TriplesMaps")

    entities = parser.get_entity_info()
    print(f"Found {len(entities)} distinct entity types")

    return True


def test_r2rml_parser_npd():
    """Test the generic R2RML parser on NRG."""
    print("\n" + "=" * 60)
    print("TEST 3: R2RML Parser (NRG)")
    print("=" * 60)

    mapping_path = _get_mapping_path("nrg")
    if not mapping_path:
        print("ERROR: NRG mapping not found")
        return False

    parser = R2RMLParser(mapping_path)

    print(f"\nFound {len(parser.triples_maps)} TriplesMaps")

    entities = parser.get_entity_info()
    print(f"Found {len(entities)} distinct entity types")

    return True


def test_analyze_mapping_entities():
    """Test the analyze_mapping_entities tool."""
    print("\n" + "=" * 60)
    print("TEST 4: analyze_mapping_entities Tool")
    print("=" * 60)

    for dataset in ["edu", "trn", "nrg"]:
        print(f"\n--- {dataset.upper()} ---")
        result = analyze_mapping_entities.invoke({"dataset": dataset})
        data = json.loads(result)

        if data.get("success"):
            print(f"  TriplesMaps: {data['total_triples_maps']}")
            print(f"  Entity Types: {data['entity_types']}")
        else:
            print(f"  ERROR: {data.get('error')}")
            return False

    return True


def test_analyze_iri_dependencies():
    """Test the analyze_iri_dependencies tool."""
    print("\n" + "=" * 60)
    print("TEST 5: analyze_iri_dependencies Tool")
    print("=" * 60)

    # Test finding a common variable in TRN
    print("\nTRN: Finding usages of 'trip_id'")
    result = analyze_iri_dependencies.invoke({
        "dataset": "trn",
        "variable": "trip_id"
    })
    data = json.loads(result)

    if data.get("success"):
        print(f"  Subject maps: {data['subject_map_count']}")
        print(f"  Object maps: {data['object_map_count']}")
        print(f"  Total: {data['total_usages']}")
    else:
        print(f"  ERROR: {data.get('error')}")
        return False

    # Test finding a common variable in EDU
    print("\nEDU: Finding usages of 'departmentnr'")
    result = analyze_iri_dependencies.invoke({
        "dataset": "edu",
        "variable": "departmentnr"
    })
    data = json.loads(result)

    if data.get("success"):
        print(f"  Subject maps: {data['subject_map_count']}")
        print(f"  Object maps: {data['object_map_count']}")
        print(f"  Total: {data['total_usages']}")
    else:
        print(f"  ERROR: {data.get('error')}")
        return False

    return True


def test_template_variable_extraction():
    """Test that template variable extraction works correctly."""
    print("\n" + "=" * 60)
    print("TEST 6: Template Variable Extraction")
    print("=" * 60)

    mapping_path = _get_mapping_path("edu")
    parser = R2RMLParser(mapping_path)

    test_cases = [
        ("http://example.org/{id}", ["id"]),
        ("http://example.org/{dept}/{name}", ["dept", "name"]),
        ("http://www.department{departmentnr}.university{universitynr}.edu", ["departmentnr", "universitynr"]),
        ("http://example.org/static/path", []),
    ]

    all_passed = True
    for template, expected in test_cases:
        result = parser.extract_template_variables(template)
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {template[:50]}...")
        print(f"         Expected: {expected}, Got: {result}")
        if not passed:
            all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# Testing Generic R2RML IRI Optimization Tools")
    print("#" * 60)

    results = []

    results.append(("R2RML Parser (EDU)", test_r2rml_parser_edu()))
    results.append(("R2RML Parser (TRN)", test_r2rml_parser_trn()))
    results.append(("R2RML Parser (NRG)", test_r2rml_parser_npd()))
    results.append(("analyze_mapping_entities", test_analyze_mapping_entities()))
    results.append(("analyze_iri_dependencies", test_analyze_iri_dependencies()))
    results.append(("Template Variable Extraction", test_template_variable_extraction()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
