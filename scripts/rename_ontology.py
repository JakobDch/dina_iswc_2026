"""
Ontology Decontamination Script.

Renames classes, properties, prefixes, and namespace URIs across all
ontology files, R2RML mappings, TTL semantic models, and SPARQL ground truth
queries. This prevents LLM memorization of well-known ontology mappings
(EDU, TRN, NRG) during experiments.

Usage:
    python scripts/rename_ontology.py [--dry-run] [--verbose]
"""

import argparse
import json
import os
import re
import shutil
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_mapping(mapping_path=None):
    mapping_path = Path(mapping_path) if mapping_path else PROJECT_ROOT / "scripts" / "rename_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_replacements(ds_config):
    """Build ordered list of (compiled_regex, replacement_string) tuples.

    Order matters:
      1. Specific full URI names (longest name first)
      2. Full namespace URI (with #)
      3. Base URI without # (with lookahead to avoid matching longer URIs)
      4. Specific prefixed names (longest name first)
      5. Generic prefix replacement
    """
    old_ns = ds_config["old_namespace"]
    new_ns = ds_config["new_namespace"]
    old_prefix = ds_config["old_prefix"]
    new_prefix = ds_config["new_prefix"]

    all_names = {
        **ds_config.get("classes", {}),
        **ds_config.get("properties", {}),
    }
    # Sort by name length descending to avoid partial matches
    sorted_names = sorted(all_names.items(), key=lambda x: len(x[0]), reverse=True)

    replacements = []

    # 1. Specific full URI names: e.g. ...owl#Student -> ...education#Learner
    #    Negative lookahead prevents matching ...owl#StudentSomething
    for old_name, new_name in sorted_names:
        pattern = re.compile(
            re.escape(old_ns + old_name) + r"(?![a-zA-Z0-9_])"
        )
        replacements.append((pattern, new_ns + new_name))

    # 2. Full namespace URI (with #): catches all remaining full-URI terms
    pattern = re.compile(re.escape(old_ns))
    replacements.append((pattern, new_ns))

    # 3. Base URI without # (for xml:base, rdf:about on ontology IRI)
    #    Lookahead ensures we don't match longer URIs like nrg-v2-ptl
    old_base = old_ns.rstrip("#")
    new_base = new_ns.rstrip("#")
    pattern = re.compile(re.escape(old_base) + r"""(?=[#"'\s>]|$)""")
    replacements.append((pattern, new_base))

    # 4. Specific prefixed names: e.g. eduo:Student -> eduo:Learner
    #    Negative lookahead prevents matching eduo:StudentSomething
    for old_name, new_name in sorted_names:
        pattern = re.compile(
            re.escape(f"{old_prefix}:{old_name}") + r"(?![a-zA-Z0-9_])"
        )
        replacements.append((pattern, f"{new_prefix}:{new_name}"))

    # 5. Generic prefix: catches all remaining prefixed terms (eduo: -> eduo:)
    pattern = re.compile(re.escape(f"{old_prefix}:"))
    replacements.append((pattern, f"{new_prefix}:"))

    return replacements


def apply_replacements(text, replacements):
    for pattern, replacement in replacements:
        text = pattern.sub(replacement, text)
    return text


def process_file(filepath, all_replacements, verbose=False):
    """Apply all dataset replacements to a single file.

    Returns True if the file was modified.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original = f.read()
    except (UnicodeDecodeError, FileNotFoundError) as e:
        print(f"  SKIP {filepath}: {e}")
        return False

    modified = original
    for replacements in all_replacements:
        modified = apply_replacements(modified, replacements)

    if original != modified:
        if verbose:
            # Count changes
            diff_count = sum(
                1 for a, b in zip(original, modified) if a != b
            )
            print(f"  MODIFIED: {filepath} (~{diff_count} char diffs)")
        else:
            print(f"  MODIFIED: {filepath}")
        return True, modified
    return False, original


def collect_files(ds_name):
    """Collect all files that need processing for a dataset."""
    files = []

    # 1. TTL semantic models
    sm_dir = PROJECT_ROOT / "data" / "datasets" / ds_name / "class_semantic_models"
    if sm_dir.exists():
        files.extend(sorted(sm_dir.glob("*.ttl")))

    # 2. Mapping directories (base + small + large variants)
    ds_lower = ds_name.lower()
    for variant in [ds_lower, f"{ds_lower}-small", f"{ds_lower}-large"]:
        mapping_dir = PROJECT_ROOT / "mappings" / variant
        if mapping_dir.exists():
            files.extend(sorted(mapping_dir.glob("*.owl")))
            files.extend(sorted(mapping_dir.glob("*.ttl")))

    # 3. NRG extra mapping file
    if ds_name == "NRG":
        extra = PROJECT_ROOT / "data" / "datasets" / "NRG" / "nrg_dataset_mapping.r2rml"
        if extra.exists():
            files.append(extra)

    return files


def rename_ttl_files(ds_name, class_mappings):
    """Rename TTL files for renamed classes."""
    sm_dir = PROJECT_ROOT / "data" / "datasets" / ds_name / "class_semantic_models"
    if not sm_dir.exists():
        return

    for old_name, new_name in class_mappings.items():
        old_path = sm_dir / f"{old_name}.ttl"
        new_path = sm_dir / f"{new_name}.ttl"
        if old_path.exists():
            os.rename(old_path, new_path)
            print(f"  RENAMED: {old_name}.ttl -> {new_name}.ttl")


def process_experimental_corpus(all_replacements_by_ds):
    """Process the experimental corpus Python file."""
    corpus_path = PROJECT_ROOT / "data" / "queries" / "experimental_corpus.py"
    if not corpus_path.exists():
        print(f"  WARNING: {corpus_path} not found!")
        return

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    original = text
    for ds_name, replacements in all_replacements_by_ds.items():
        text = apply_replacements(text, replacements)

    if text != original:
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  MODIFIED: {corpus_path}")
    else:
        print(f"  UNCHANGED: {corpus_path}")


def process_shared_files(all_replacements_by_ds):
    """Process top-level shared mapping/ontology files."""
    shared_files = [
        PROJECT_ROOT / "mappings" / "ontology.owl",
        PROJECT_ROOT / "mappings" / "mapping.ttl",
    ]

    for filepath in shared_files:
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        original = text
        for ds_name, replacements in all_replacements_by_ds.items():
            text = apply_replacements(text, replacements)

        if text != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  MODIFIED: {filepath}")


def clean_caches():
    """Delete embedding caches and ground truth cache."""
    cache_dirs = [
        PROJECT_ROOT / "data" / "cache" / "embeddings",
    ]
    cache_files = [
        PROJECT_ROOT / "data" / "cache" / "ground_truth_full.json",
    ]

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  DELETED: {cache_dir}")

    for cache_file in cache_files:
        if cache_file.exists():
            os.remove(cache_file)
            print(f"  DELETED: {cache_file}")


def validate_no_old_references(mapping):
    """Check that no old namespace URIs remain in processed files."""
    print("\n" + "=" * 60)
    print("VALIDATION: Checking for remaining old references")
    print("=" * 60)

    issues = []
    for ds_name, ds_config in mapping["datasets"].items():
        old_ns = ds_config["old_namespace"]
        old_prefix_pattern = ds_config["old_prefix"] + ":"

        # Check semantic models
        sm_dir = PROJECT_ROOT / "data" / "datasets" / ds_name / "class_semantic_models"
        if sm_dir.exists():
            for ttl_file in sm_dir.glob("*.ttl"):
                with open(ttl_file, "r", encoding="utf-8") as f:
                    content = f.read()
                if old_ns in content:
                    issues.append(f"  OLD NAMESPACE in {ttl_file}")
                if old_prefix_pattern in content:
                    issues.append(f"  OLD PREFIX '{old_prefix_pattern}' in {ttl_file}")

        # Check mapping dirs
        ds_lower = ds_name.lower()
        for variant in [ds_lower, f"{ds_lower}-small", f"{ds_lower}-large"]:
            mapping_dir = PROJECT_ROOT / "mappings" / variant
            if mapping_dir.exists():
                for f_path in list(mapping_dir.glob("*.owl")) + list(mapping_dir.glob("*.ttl")):
                    with open(f_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if old_ns in content:
                        issues.append(f"  OLD NAMESPACE in {f_path}")
                    if old_prefix_pattern in content:
                        issues.append(f"  OLD PREFIX '{old_prefix_pattern}' in {f_path}")

    # Check experimental corpus
    corpus_path = PROJECT_ROOT / "data" / "queries" / "experimental_corpus.py"
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            content = f.read()
        for ds_name, ds_config in mapping["datasets"].items():
            old_ns = ds_config["old_namespace"]
            old_prefix_pattern = ds_config["old_prefix"] + ":"
            if old_ns in content:
                issues.append(f"  OLD NAMESPACE '{old_ns}' in {corpus_path}")
            if old_prefix_pattern in content:
                issues.append(f"  OLD PREFIX '{old_prefix_pattern}' in {corpus_path}")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(issue)
        return False
    else:
        print("ALL CLEAR - no old references found.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Rename ontology classes/properties for decontamination")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--mapping", default=None, help="Path to a rename mapping JSON (default: scripts/rename_mapping.json)")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)

    # Build replacements for each dataset
    all_replacements = {}
    for ds_name, ds_config in mapping["datasets"].items():
        all_replacements[ds_name] = build_replacements(ds_config)

    # Process dataset-specific files
    for ds_name, ds_config in mapping["datasets"].items():
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {ds_name}")
        print(f"  Prefix: {ds_config['old_prefix']}: -> {ds_config['new_prefix']}:")
        print(f"  Namespace: {ds_config['old_namespace']}")
        print(f"         ->  {ds_config['new_namespace']}")
        print(f"{'=' * 60}")

        replacements = all_replacements[ds_name]
        files = collect_files(ds_name)

        print(f"\n  Found {len(files)} files to process")

        for filepath in files:
            was_modified, new_content = process_file(
                filepath, [replacements], verbose=args.verbose
            )
            if was_modified and not args.dry_run:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)

        # Rename TTL files (must happen AFTER content replacement)
        if not args.dry_run:
            print(f"\n  Renaming TTL files:")
            rename_ttl_files(ds_name, ds_config.get("classes", {}))

    # Process shared files (top-level mappings/ontology)
    print(f"\n{'=' * 60}")
    print("Processing shared files")
    print(f"{'=' * 60}")
    if not args.dry_run:
        process_shared_files(all_replacements)

    # Process experimental corpus (contains queries for all datasets)
    print(f"\n{'=' * 60}")
    print("Processing experimental corpus")
    print(f"{'=' * 60}")
    if not args.dry_run:
        process_experimental_corpus(all_replacements)

    # Clean caches
    print(f"\n{'=' * 60}")
    print("Cleaning caches")
    print(f"{'=' * 60}")
    if not args.dry_run:
        clean_caches()
    else:
        print("  (skipped in dry-run mode)")

    # Validation
    if not args.dry_run:
        valid = validate_no_old_references(mapping)
        if not valid:
            print("\nWARNING: Some old references remain! Check the issues above.")
            return 1

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print("DRY RUN complete. No files were modified.")
    else:
        print("DONE. All files have been updated.")
        print("\nNext steps:")
        print("  1. Restart Docker containers (OnTop needs new mappings)")
        print("  2. Rebuild FAISS index:  python scripts/preindex_embeddings.py --force")
        print("  3. Regenerate GT cache:  python scripts/extract_ground_truth_cache.py")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())
