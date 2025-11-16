"""
Full experiment runner: 100 tests for both approaches.

This script:
1. Generates 100 sets of credit facility parameters
2. Fetches distractor documents (subscription agreements)
3. Runs both TaleStudio and Report System approaches
4. Evaluates results with LLM judge
5. Generates comparative analysis
"""

import os
import json
import asyncio
from typing import Dict, List, Any
from pathlib import Path
import sys
from datetime import datetime
import statistics

sys.path.append(str(Path(__file__).parent.parent))

from test_talestudio_approach import TaleStudioApproach
from test_report_approach import ReportSystemApproach
from evaluate_results import LLMJudge
from utils.dummy_data import generate_batch
from utils.edgar_fetcher import EdgarFetcher


class ExperimentRunner:
    """Orchestrates the full experiment."""

    def __init__(self, api_key: str, num_tests: int = 100):
        self.api_key = api_key
        self.num_tests = num_tests
        self.output_dir = Path("experiments/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def run(self):
        """Run the full experiment."""
        print(f"\n{'='*80}")
        print(f"CREDIT FACILITY GENERATION EXPERIMENT")
        print(f"TaleStudio (Semantic Retrieval) vs Report System (Full Context)")
        print(f"Tests: {self.num_tests}")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'='*80}\n")

        # Step 1: Prepare test data
        print("Step 1: Preparing test data...")
        params_batch = self._generate_test_parameters()
        distractor_docs = self._get_distractor_documents()
        print(f"  ✓ Generated {len(params_batch)} test cases")
        print(f"  ✓ Fetched {len(distractor_docs)} distractor documents")

        # Step 2: Run TaleStudio approach
        print("\nStep 2: Running TaleStudio approach...")
        talestudio_results = await self._run_talestudio_tests(
            params_batch,
            distractor_docs
        )
        self._save_results(talestudio_results, "talestudio")
        print(f"  ✓ Completed {len(talestudio_results)} tests")

        # Step 3: Run Report System approach
        print("\nStep 3: Running Report System approach...")
        report_results = await self._run_report_system_tests(
            params_batch,
            distractor_docs
        )
        self._save_results(report_results, "report_system")
        print(f"  ✓ Completed {len(report_results)} tests")

        # Step 4: Evaluate results
        print("\nStep 4: Evaluating results...")
        talestudio_evals = await self._evaluate_results(
            talestudio_results,
            "talestudio"
        )
        report_evals = await self._evaluate_results(
            report_results,
            "report_system"
        )
        print(f"  ✓ Completed evaluations")

        # Step 5: Generate comparative analysis
        print("\nStep 5: Generating comparative analysis...")
        self._generate_analysis(talestudio_evals, report_evals)
        print(f"  ✓ Analysis complete")

        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")

    def _generate_test_parameters(self) -> List[Dict]:
        """Generate test parameters."""
        return generate_batch(num_samples=self.num_tests, start_seed=1000)

    def _get_distractor_documents(self) -> List[str]:
        """Get distractor documents."""
        fetcher = EdgarFetcher()
        return fetcher.get_sample_subscription_agreements(10)

    async def _run_talestudio_tests(
        self,
        params_batch: List[Dict],
        distractor_docs: List[str]
    ) -> List[Dict]:
        """Run all TaleStudio tests."""
        approach = TaleStudioApproach(self.api_key)
        results = []

        for i, params in enumerate(params_batch):
            print(f"  Test {i+1}/{len(params_batch)}: {params['borrower_name'][:30]}...")

            try:
                result = await approach.generate_credit_facility(
                    params=params,
                    distractor_docs=distractor_docs
                )

                result["test_id"] = i
                result["params"] = params
                result["distractor_count"] = len(distractor_docs)
                results.append(result)

                # Rate limiting
                await asyncio.sleep(2)

            except Exception as e:
                print(f"    ✗ Error: {e}")
                results.append({
                    "test_id": i,
                    "params": params,
                    "error": str(e),
                    "agreement_text": "",
                    "metadata": {"approach": "talestudio", "error": True}
                })

        return results

    async def _run_report_system_tests(
        self,
        params_batch: List[Dict],
        distractor_docs: List[str]
    ) -> List[Dict]:
        """Run all Report System tests."""
        approach = ReportSystemApproach(self.api_key)
        results = []

        for i, params in enumerate(params_batch):
            print(f"  Test {i+1}/{len(params_batch)}: {params['borrower_name'][:30]}...")

            try:
                result = await approach.generate_credit_facility(
                    params=params,
                    distractor_docs=distractor_docs
                )

                result["test_id"] = i
                result["params"] = params
                result["distractor_count"] = len(distractor_docs)
                results.append(result)

                # Rate limiting
                await asyncio.sleep(2)

            except Exception as e:
                print(f"    ✗ Error: {e}")
                results.append({
                    "test_id": i,
                    "params": params,
                    "error": str(e),
                    "agreement_text": "",
                    "metadata": {"approach": "report_system", "error": True}
                })

        return results

    def _save_results(self, results: List[Dict], approach: str):
        """Save results to JSONL file."""
        output_file = self.output_dir / f"{approach}_results_{self.timestamp}.jsonl"

        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, default=str) + "\n")

        print(f"    Saved to: {output_file}")

    async def _evaluate_results(
        self,
        results: List[Dict],
        approach: str
    ) -> List[Dict]:
        """Evaluate results with LLM judge."""
        judge = LLMJudge(self.api_key)
        evaluations = []

        for i, result in enumerate(results):
            if result.get("error"):
                print(f"    Skipping test {i} (error)")
                continue

            print(f"    Evaluating {i+1}/{len(results)}...")

            try:
                evaluation = await judge.evaluate(
                    agreement_text=result['agreement_text'],
                    params=result['params'],
                    distractor_docs=[]
                )

                evaluations.append({
                    "test_id": result['test_id'],
                    "approach": approach,
                    "evaluation": evaluation,
                    "metadata": result['metadata']
                })

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                print(f"      ✗ Evaluation error: {e}")

        # Save evaluations
        eval_file = self.output_dir / f"{approach}_evaluations_{self.timestamp}.jsonl"
        with open(eval_file, 'w') as f:
            for eval in evaluations:
                f.write(json.dumps(eval, indent=2) + "\n")

        return evaluations

    def _generate_analysis(
        self,
        talestudio_evals: List[Dict],
        report_evals: List[Dict]
    ):
        """Generate comparative analysis."""

        analysis = {
            "timestamp": self.timestamp,
            "num_tests": self.num_tests,
            "talestudio": self._calculate_stats(talestudio_evals),
            "report_system": self._calculate_stats(report_evals),
        }

        # Calculate winner for each metric
        analysis["comparison"] = {
            "overall_winner": "talestudio" if analysis["talestudio"]["overall_score"]["mean"] > analysis["report_system"]["overall_score"]["mean"] else "report_system",
            "hallucination_winner": "talestudio" if analysis["talestudio"]["hallucination"]["mean"] < analysis["report_system"]["hallucination"]["mean"] else "report_system",
            "task_adherence_winner": "talestudio" if analysis["talestudio"]["task_adherence"]["mean"] > analysis["report_system"]["task_adherence"]["mean"] else "report_system",
            "completeness_winner": "talestudio" if analysis["talestudio"]["completeness"]["mean"] > analysis["report_system"]["completeness"]["mean"] else "report_system",
            "quality_winner": "talestudio" if analysis["talestudio"]["quality"]["mean"] > analysis["report_system"]["quality"]["mean"] else "report_system",
            "precision_winner": "talestudio" if analysis["talestudio"]["precision"]["mean"] > analysis["report_system"]["precision"]["mean"] else "report_system",
        }

        # Save analysis
        analysis_file = self.output_dir / f"comparative_analysis_{self.timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Print summary
        self._print_summary(analysis)

        return analysis

    def _calculate_stats(self, evaluations: List[Dict]) -> Dict:
        """Calculate statistics for evaluations."""
        if not evaluations:
            return {}

        stats = {}

        for metric in ["overall_score", "hallucination", "task_adherence", "completeness", "quality", "precision"]:
            if metric == "hallucination":
                scores = [e['evaluation'][metric]['score'] for e in evaluations]
            elif metric == "overall_score":
                scores = [e['evaluation'][metric] for e in evaluations]
            else:
                scores = [e['evaluation'][metric]['score'] for e in evaluations]

            stats[metric] = {
                "mean": statistics.mean(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
                "median": statistics.median(scores)
            }

        return stats

    def _print_summary(self, analysis: Dict):
        """Print comparative summary."""
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS SUMMARY")
        print(f"{'='*80}\n")

        metrics = [
            ("Overall Score", "overall_score", False),
            ("Hallucination", "hallucination", True),  # Lower is better
            ("Task Adherence", "task_adherence", False),
            ("Completeness", "completeness", False),
            ("Quality", "quality", False),
            ("Precision", "precision", False),
        ]

        for metric_name, metric_key, lower_is_better in metrics:
            ts_score = analysis["talestudio"][metric_key]["mean"]
            rs_score = analysis["report_system"][metric_key]["mean"]

            if lower_is_better:
                winner = "TaleStudio" if ts_score < rs_score else "Report System"
                diff = rs_score - ts_score
            else:
                winner = "TaleStudio" if ts_score > rs_score else "Report System"
                diff = ts_score - rs_score

            print(f"{metric_name}:")
            print(f"  TaleStudio:     {ts_score:.2f} ± {analysis['talestudio'][metric_key]['stdev']:.2f}")
            print(f"  Report System:  {rs_score:.2f} ± {analysis['report_system'][metric_key]['stdev']:.2f}")
            print(f"  Winner:         {winner} ({'better' if not lower_is_better else 'lower'} by {abs(diff):.2f})")
            print()

        print(f"{'='*80}\n")


async def main():
    """Main entry point."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run credit facility generation experiment")
    parser.add_argument("--num-tests", type=int, default=100, help="Number of tests to run (default: 100)")
    parser.add_argument("--quick", action="store_true", help="Quick test with only 5 samples")
    args = parser.parse_args()

    num_tests = 5 if args.quick else args.num_tests

    runner = ExperimentRunner(api_key=api_key, num_tests=num_tests)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
