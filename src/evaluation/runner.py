"""
Experiment runner for benchmarking different approaches.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from src.config import get_settings, ApproachType, RESULTS_DIR
from src.models.query import NaturalLanguageQuery
from src.models.evaluation import (
    SingleRunResult,
    QueryAggregatedResult,
    ExperimentResult,
    ResultMetrics,
    QueryMetrics,
    AgentMetrics,
    TokenUsage,
)
from src.baseline.metrics import calculate_result_metrics, calculate_edit_cost
from src.baseline.query_correction import correct_query_iteratively
from src.baseline.llm_services import create_llm_instance, LLMInstance
from src.baseline.config import get_settings as get_baseline_settings, LLM_PROFILES
from src.endpoints.ontop import OnTopEndpoint
from src.agents.orchestrator import OrchestratorAgent

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runner for executing and evaluating experiments."""

    def __init__(
        self,
        experiment_name: str,
        approaches: list[ApproachType] | None = None,
        llm_models: list[str] | None = None,
        runs_per_query: int = 5,
        correction_llm_profile: str | None = None,
        correction_max_iterations: int = 5,
    ):
        """
        Initialize the experiment runner.

        Args:
            experiment_name: Name of the experiment
            approaches: List of approaches to test
            llm_models: List of LLM models to use
            runs_per_query: Number of runs per query for statistical significance
            correction_llm_profile: LLM profile to use for query correction (default from settings)
            correction_max_iterations: Max iterations for iterative query correction
        """
        self.experiment_name = experiment_name
        self.approaches = approaches or ["baseline", "agentic_grep", "agentic_semantic"]
        self.llm_models = llm_models or ["gpt-4o"]
        self.runs_per_query = runs_per_query
        self.correction_max_iterations = correction_max_iterations

        self.results: list[QueryAggregatedResult] = []
        self.total_token_usage = TokenUsage()

        # Initialize SPARQL endpoint for query execution
        baseline_settings = get_baseline_settings()
        self.sparql_endpoint = OnTopEndpoint(baseline_settings.ontop_sparql_url)

        # Initialize LLM for query correction (used for edit cost evaluation)
        correction_profile = correction_llm_profile or baseline_settings.sparql_edit_evaluation_llm_profile
        try:
            self.correction_llm: LLMInstance | None = create_llm_instance(
                profile_key=correction_profile,
                llm_profiles=LLM_PROFILES,
                settings=baseline_settings,
            )
            logger.info(f"Initialized correction LLM with profile: {correction_profile}")
        except Exception as e:
            logger.warning(f"Failed to initialize correction LLM: {e}. Will use groundtruth directly.")
            self.correction_llm = None

    async def run_single(
        self,
        query: NaturalLanguageQuery,
        approach: ApproachType,
        llm_model: str,
        run_number: int,
    ) -> SingleRunResult:
        """
        Run a single experiment iteration.

        Args:
            query: The query to process
            approach: Which approach to use
            llm_model: Which LLM model to use
            run_number: Run number (1-indexed)

        Returns:
            SingleRunResult with all metrics
        """
        start_time = datetime.now()
        generated_sparql = ""
        final_results: list[dict] = []
        success = False
        error_message = None
        agent_metrics = None

        try:
            if approach == "baseline":
                # TODO: Implement baseline pipeline
                generated_sparql = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
                success = True

            elif approach in ["agentic_grep", "agentic_semantic"]:
                agent = OrchestratorAgent()
                state = await agent.run(
                    user_query=query.text,
                    approach=approach,  # type: ignore
                )

                generated_sparql = state.get("final_query", "")
                final_results = [state.get("final_results", {})]
                success = generated_sparql is not None

                agent_metrics = AgentMetrics(
                    iteration_count=state.get("iteration_count", 0),
                    confidence_score=state.get("confidence_score", 0.0),
                )

        except Exception as e:
            logger.error(f"Error in run: {e}")
            error_message = str(e)

        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        # Calculate metrics if ground truth is available
        result_metrics = None
        query_metrics = None

        if query.expected_results and final_results:
            metrics = calculate_result_metrics(
                final_results[0] if final_results else {},
                query.expected_results[0] if query.expected_results else {},
            )
            result_metrics = ResultMetrics(
                execution_accuracy=metrics["execution_accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
            )

        if query.expected_sparql and generated_sparql:
            # Use iterative agent-based correction with validation
            corrected_query = query.expected_sparql  # Default to groundtruth
            correction_success = False
            correction_iterations = 0
            correction_attempts: list[dict] = []

            if self.correction_llm is not None:
                try:
                    # First, execute groundtruth query to get expected results
                    groundtruth_result = await self.sparql_endpoint.execute_query(
                        query.expected_sparql
                    )

                    if groundtruth_result.success and groundtruth_result.results:
                        # Create execute function wrapper for iterative correction
                        async def execute_fn(q: str) -> tuple[bool, dict | None, str | None, str | None]:
                            result = await self.sparql_endpoint.execute_query(q)
                            return (
                                result.success,
                                result.results,
                                result.error_type,
                                result.error_message,
                            )

                        # Run iterative correction with validation
                        correction_result = await correct_query_iteratively(
                            generated_query=generated_sparql,
                            groundtruth_query=query.expected_sparql,
                            groundtruth_results=groundtruth_result.results,
                            llm_instance=self.correction_llm,
                            execute_fn=execute_fn,
                            user_query=query.text,
                            max_iterations=self.correction_max_iterations,
                            request_id=f"{query.id}_{run_number}",
                        )

                        corrected_query = correction_result.get(
                            "corrected_query", query.expected_sparql
                        )
                        correction_success = correction_result.get("success", False)
                        correction_iterations = correction_result.get("iterations", 0)
                        correction_attempts = correction_result.get("attempts", [])

                        # Track token usage from all correction iterations
                        if "token_usage" in correction_result:
                            usage = correction_result["token_usage"]
                            self.total_token_usage.prompt_tokens += usage.get("input_tokens", 0)
                            self.total_token_usage.completion_tokens += usage.get("output_tokens", 0)
                            self.total_token_usage.total_tokens += usage.get("total_tokens", 0)
                    else:
                        logger.warning(
                            f"Could not execute groundtruth query: {groundtruth_result.error_message}"
                        )

                except Exception as e:
                    logger.warning(f"Query correction failed: {e}. Using groundtruth.")

            # Calculate edit cost using algebraic analysis
            edit_result = calculate_edit_cost(
                generated_query=generated_sparql,
                corrected_query=corrected_query,
                normalize_variables=True,
                request_id=f"{query.id}_{run_number}",
            )

            query_metrics = QueryMetrics(
                edit_cost=edit_result.get("normalized_score", 0.0),
                actual_edit_cost=edit_result.get("actual_cost", 0),
                max_edit_cost=edit_result.get("max_cost", 0),
                edit_details=edit_result.get("edit_details", []),
                corrected_query=corrected_query,
                syntax_valid=success,
                correction_success=correction_success,
                correction_iterations=correction_iterations,
                correction_attempts=correction_attempts,
            )

        return SingleRunResult(
            query_id=query.id,
            approach=approach,
            llm_model=llm_model,
            run_number=run_number,
            generated_sparql=generated_sparql,
            final_results=final_results,
            result_metrics=result_metrics,
            query_metrics=query_metrics,
            agent_metrics=agent_metrics,
            total_time_ms=total_time_ms,
            success=success,
            error_message=error_message,
        )

    async def run_query(
        self,
        query: NaturalLanguageQuery,
        approach: ApproachType,
        llm_model: str,
    ) -> QueryAggregatedResult:
        """
        Run all iterations for a single query/approach/model combination.
        """
        runs: list[SingleRunResult] = []

        for run_num in range(1, self.runs_per_query + 1):
            result = await self.run_single(query, approach, llm_model, run_num)
            runs.append(result)

        # Aggregate metrics
        successful_runs = [r for r in runs if r.success]
        success_rate = len(successful_runs) / len(runs) if runs else 0.0

        # Calculate mean and std for metrics
        exec_accuracies = [
            r.result_metrics.execution_accuracy
            for r in runs
            if r.result_metrics
        ]
        f1_scores = [
            r.result_metrics.f1_score
            for r in runs
            if r.result_metrics
        ]
        edit_costs = [
            r.query_metrics.edit_cost
            for r in runs
            if r.query_metrics
        ]

        def mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def std(values: list[float]) -> float:
            if len(values) < 2:
                return 0.0
            m = mean(values)
            variance = sum((x - m) ** 2 for x in values) / len(values)
            return variance ** 0.5

        return QueryAggregatedResult(
            query_id=query.id,
            query_text=query.text,
            approach=approach,
            llm_model=llm_model,
            num_runs=len(runs),
            mean_execution_accuracy=mean(exec_accuracies),
            std_execution_accuracy=std(exec_accuracies),
            mean_f1_score=mean(f1_scores),
            std_f1_score=std(f1_scores),
            mean_edit_cost=mean(edit_costs),
            std_edit_cost=std(edit_costs),
            success_rate=success_rate,
            runs=runs,
        )

    async def run_experiment(
        self,
        queries: list[NaturalLanguageQuery],
    ) -> ExperimentResult:
        """
        Run the full experiment across all queries, approaches, and models.
        """
        logger.info(
            f"Starting experiment '{self.experiment_name}' with "
            f"{len(queries)} queries, {len(self.approaches)} approaches, "
            f"{len(self.llm_models)} models, {self.runs_per_query} runs each"
        )

        total_iterations = (
            len(queries) * len(self.approaches) * len(self.llm_models)
        )

        with tqdm(total=total_iterations, desc="Running experiment") as pbar:
            for query in queries:
                for approach in self.approaches:
                    for llm_model in self.llm_models:
                        result = await self.run_query(query, approach, llm_model)
                        self.results.append(result)
                        pbar.update(1)

        return ExperimentResult(
            experiment_name=self.experiment_name,
            config={
                "approaches": self.approaches,
                "llm_models": self.llm_models,
                "runs_per_query": self.runs_per_query,
            },
            results=self.results,
            total_queries=len(queries),
            total_runs=len(queries) * len(self.approaches) * len(self.llm_models) * self.runs_per_query,
            total_token_usage=self.total_token_usage,
        )

    def save_results(self, result: ExperimentResult, output_dir: Path | None = None) -> Path:
        """Save experiment results to JSON file."""
        output_dir = output_dir or RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")
        return output_path


async def run_experiment(
    experiment_name: str,
    queries: list[NaturalLanguageQuery],
    approaches: list[ApproachType] | None = None,
    llm_models: list[str] | None = None,
    runs_per_query: int = 5,
    save_results: bool = True,
    correction_llm_profile: str | None = None,
    correction_max_iterations: int = 5,
) -> ExperimentResult:
    """
    Convenience function to run an experiment.

    Args:
        experiment_name: Name of the experiment
        queries: List of queries to test
        approaches: Approaches to compare
        llm_models: LLM models to use
        runs_per_query: Runs per query for statistics
        save_results: Whether to save results to disk
        correction_llm_profile: LLM profile for query correction (default from settings)
        correction_max_iterations: Max iterations for iterative query correction

    Returns:
        ExperimentResult with all data
    """
    runner = ExperimentRunner(
        experiment_name=experiment_name,
        approaches=approaches,
        llm_models=llm_models,
        runs_per_query=runs_per_query,
        correction_llm_profile=correction_llm_profile,
        correction_max_iterations=correction_max_iterations,
    )

    result = await runner.run_experiment(queries)

    if save_results:
        runner.save_results(result)

    return result
