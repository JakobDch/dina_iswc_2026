"""
Example script to run an experiment.

Usage:
    python run_experiment.py
"""

import asyncio
import logging
from pathlib import Path

from src.config import get_settings
from src.models.query import NaturalLanguageQuery
from src.evaluation.runner import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Example queries (replace with actual experiment queries)
EXAMPLE_QUERIES = [
    NaturalLanguageQuery(
        id="q1",
        text="Find all graduate students and their advisors",
        expected_sparql="""
            PREFIX eduo: <http://example.org/ontology/education#>
            SELECT ?student ?advisor
            WHERE {
                ?student a eduo:DoctoralCandidate .
                ?student eduo:advisor ?advisor .
            }
        """,
    ),
    NaturalLanguageQuery(
        id="q2",
        text="Which departments are sub-organizations of which universities?",
        expected_sparql="""
            PREFIX eduo: <http://example.org/ontology/education#>
            SELECT ?dept ?uni
            WHERE {
                ?dept a eduo:Department .
                ?dept eduo:subOrganizationOf ?uni .
                ?uni a eduo:University .
            }
        """,
    ),
    NaturalLanguageQuery(
        id="q3",
        text="List all lecturers and their research interests",
        expected_sparql="""
            PREFIX eduo: <http://example.org/ontology/education#>
            SELECT ?lecturer ?interest
            WHERE {
                ?lecturer a eduo:Lecturer .
                ?lecturer eduo:researchInterest ?interest .
            }
        """,
    ),
]


async def main():
    """Run the experiment."""
    settings = get_settings()

    logger.info("Starting experiment...")
    logger.info(f"OnTop endpoint: {settings.ontop_sparql_url}")

    # Run experiment with all approaches
    result = await run_experiment(
        experiment_name="iswc_2026_pilot",
        queries=EXAMPLE_QUERIES,
        approaches=["baseline", "agentic_grep", "agentic_semantic"],
        llm_models=["gpt-4o"],
        runs_per_query=3,  # Reduced for pilot
        save_results=True,
    )

    # Print summary
    from src.evaluation.metrics import calculate_metrics, generate_latex_table

    metrics = calculate_metrics(result)

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    for approach, data in metrics["by_approach"].items():
        print(f"\n{approach}:")
        print(f"  Execution Accuracy: {data['execution_accuracy']['mean']:.3f} ± {data['execution_accuracy']['std']:.3f}")
        print(f"  F1 Score: {data['f1_score']['mean']:.3f} ± {data['f1_score']['std']:.3f}")
        print(f"  Success Rate: {data['success_rate']['mean']:.3f}")

    print("\n" + "=" * 60)
    print("LaTeX Table:")
    print("=" * 60)
    print(generate_latex_table(result))


if __name__ == "__main__":
    asyncio.run(main())
