#!/usr/bin/env python3
"""
Starts all experiment containers in a controlled manner to avoid overloading Docker.

Usage:
    python scripts/start_containers.py [--slots N]
"""

import argparse
import subprocess
import time
import sys


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_container_status(name: str) -> str:
    """Get container status (running, exited, etc.)."""
    result = run_cmd(["docker", "inspect", "-f", "{{.State.Status}}", name], check=False)
    return result.stdout.strip() if result.returncode == 0 else "not_found"


def wait_for_healthy(name: str, timeout: int = 120) -> bool:
    """Wait for a container to become healthy."""
    start = time.time()
    while time.time() - start < timeout:
        result = run_cmd(["docker", "inspect", "-f", "{{.State.Health.Status}}", name], check=False)
        status = result.stdout.strip()
        if status == "healthy":
            return True
        if status == "" or "unhealthy" in status:
            container_status = get_container_status(name)
            if container_status == "running":
                return True
            elif container_status == "exited":
                return False
        time.sleep(2)
    return False


def start_container(name: str) -> bool:
    """Start a container if not running."""
    status = get_container_status(name)
    if status == "running":
        print(f"  {name}: already running")
        return True

    result = run_cmd(["docker", "start", name], check=False)
    if result.returncode != 0:
        print(f"  {name}: FAILED - {result.stderr.strip()}")
        return False

    print(f"  {name}: started")
    return True


def main():
    parser = argparse.ArgumentParser(description="Start experiment containers")
    parser.add_argument("--slots", type=int, default=4, help="Number of slots (default: 4)")
    parser.add_argument("--skip-mysql", action="store_true", help="Skip MySQL container")
    args = parser.parse_args()

    datasets_small = ["edu_small", "trn_small", "nrg_small", "bsbm", "lca"]
    datasets_large = ["edu_large", "trn_large", "nrg_large"]

    # Step 1: Start MySQL
    if not args.skip_mysql:
        print("\n[1/4] Starting MySQL...")
        if not start_container("dina_mysql"):
            print("ERROR: MySQL failed to start!")
            sys.exit(1)

        print("  Waiting for MySQL to be healthy...")
        if not wait_for_healthy("dina_mysql", timeout=120):
            print("ERROR: MySQL did not become healthy!")
            sys.exit(1)
        print("  MySQL is healthy!")

    # Step 2: Start small containers (low memory)
    print(f"\n[2/4] Starting small containers ({len(datasets_small)} per slot)...")
    for slot in range(args.slots):
        print(f"\n  Slot {slot}:")
        for dataset in datasets_small:
            name = f"dina_ontop_{dataset}_slot{slot}"
            start_container(name)
        time.sleep(2)

    # Step 3: Wait for small containers
    print("\n[3/4] Waiting for small containers to be healthy...")
    for slot in range(args.slots):
        for dataset in datasets_small:
            name = f"dina_ontop_{dataset}_slot{slot}"
            if not wait_for_healthy(name, timeout=180):
                print(f"  WARNING: {name} not healthy after 180s")

    # Step 4: Start large containers (high memory)
    print(f"\n[4/4] Starting large containers ({len(datasets_large)} per slot)...")
    for slot in range(args.slots):
        print(f"\n  Slot {slot}:")
        for dataset in datasets_large:
            name = f"dina_ontop_{dataset}_slot{slot}"
            start_container(name)
        time.sleep(5)

    # Final status
    print("\n" + "=" * 60)
    print("Waiting for all containers to be healthy...")
    time.sleep(30)

    result = run_cmd(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"], check=False)
    healthy = 0
    total = 0
    for line in result.stdout.strip().split("\n"):
        if "dina_" in line:
            total += 1
            if "healthy" in line:
                healthy += 1

    print(f"\nStatus: {healthy}/{total} containers healthy")

    if healthy < total:
        print("\nContainers not yet healthy:")
        for line in result.stdout.strip().split("\n"):
            if "dina_" in line and "healthy" not in line:
                print(f"  {line}")


if __name__ == "__main__":
    main()
