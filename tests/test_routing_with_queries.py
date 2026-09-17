"""
Test Agent Routing with Experimental Corpus Queries on LARGE Datasets.

Tests the new routing logic where the SPARQL Agent decides:
1. Try a different query
2. Delegate to Mapping Optimizer
3. Delegate to Multi-Step Agent

Uses LARGE datasets to trigger actual timeouts!
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use LARGE dataset queries - these will actually timeout
from data.queries.experimental_corpus import LARGE04, LARGE08, LARGE01, LARGE05


async def test_query(nl_query: str, dataset: str, description: str):
    """Test a single query and observe routing behavior."""
    from src.agents.generation.sparql_agent import SPARQLGenerationAgent

    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"Dataset: {dataset}")
    print(f"NL Query: {nl_query}")
    print(f"{'='*70}")

    # Initialize agent with DeepSeek
    agent = SPARQLGenerationAgent(
        llm_model="deepseek-chat",
        enable_endpoint_selection=True,
        enable_tools=True,
        max_tool_calls=15,  # Allow more tool calls for testing
    )

    # Build schema context pointing to the LARGE dataset
    schema_context = [
        f"# Dataset: {dataset}",
        f"# Use the large dataset endpoint for this query",
        f"# This is a performance test - queries may timeout",
    ]

    try:
        print("\n[Starting generation...]")
        results, messages = await agent.generate(
            user_query=nl_query,
            schema_context=schema_context,
            k=1,
        )

        print("\n[Generation complete]")

        if results:
            result = results[0]
            print(f"\nStatus: {result.get('status', 'N/A')}")

            if result.get('status') == 'MULTI_STEP_RESULT':
                print("[ROUTING] -> Multi-Step Agent was used")
                ms_result = result.get('multi_step_result', {})
                print(f"  Success: {ms_result.get('success')}")
                print(f"  Steps: {len(ms_result.get('steps', []))}")

            elif 'can_help' in str(result):
                print("[ROUTING] -> Mapping Optimizer was considered")
                print(f"  can_help: {result.get('can_help', 'N/A')}")

            else:
                print("[ROUTING] -> Direct query generation")

            if result.get('query'):
                query_preview = result['query'][:200]
                print(f"\nGenerated Query Preview:\n{query_preview}...")

            if result.get('reasoning'):
                reasoning = result['reasoning'][:300].encode('ascii', 'replace').decode('ascii')
                print(f"\nReasoning: {reasoning}...")

            if result.get('decision_reasoning'):
                decision = result['decision_reasoning'][:300].encode('ascii', 'replace').decode('ascii')
                print(f"\nDecision: {decision}...")
        else:
            print("\n[No results returned]")

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()

    return results


async def main():
    """Run routing tests with TRN-Large and EDU-Large queries."""
    print("\n" + "="*70)
    print("AGENT ROUTING TEST - LARGE Dataset Queries")
    print("Model: DeepSeek Chat")
    print("Endpoints: trn-large (8091), edu-large (8090)")
    print("="*70)

    # Test 1: TRN-Large Query - Routes and Stops
    # StopTime table has millions of rows - will timeout and trigger routing
    print("\n\n### TEST 1: TRN-LARGE - Routes and Stops ###")
    await test_query(
        nl_query="What routes serve which stops?",
        dataset="trn-large",
        description="TRN-Large: StopTime join - likely Mapping Optimizer (self-join) or Multi-Step"
    )

    print("\n" + "-"*70)

    # Test 2: EDU-Large Query - Students taking courses
    # Large student counts - will timeout and trigger routing
    print("\n\n### TEST 2: EDU-LARGE - Students and Courses ###")
    await test_query(
        nl_query="Which students are taking courses taught by professors from their own department?",
        dataset="edu-large",
        description="EDU-Large: Complex joins - likely Multi-Step (no self-joins)"
    )

    print("\n\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())