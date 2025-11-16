"""
Test Report System approach: Full context injection.

This script tests what happens when ALL context (including irrelevant
distractor documents) is injected into the prompt. This mimics the
Report System's approach of passing all research context to the LLM.
"""

import os
import json
import asyncio
from typing import Dict, List, Any
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from openai import AsyncOpenAI


class ReportSystemApproach:
    """
    Report System-style generation: Full context injection.

    Passes ALL available documents to the LLM, regardless of relevance.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_credit_facility(
        self,
        params: Dict[str, Any],
        distractor_docs: List[str]
    ) -> Dict[str, Any]:
        """
        Generate credit facility agreement with Report System approach.

        Args:
            params: Credit facility parameters
            distractor_docs: Irrelevant documents (subscription agreements)

        Returns:
            {
                "agreement_text": str,
                "metadata": {
                    "total_docs_provided": int,
                    "context_tokens_used": int,
                    "approach": "report_system"
                }
            }
        """
        # Step 1: Concatenate ALL documents into context
        # KEY DIFFERENCE: No filtering, no semantic search, just dump everything
        full_context = self._concatenate_all_context(distractor_docs)

        print(f"  Total context size: {len(full_context.split())} words")
        print(f"  Documents included: {len(distractor_docs)} (all provided, no filtering)")

        # Step 2: Create prompt with ALL context
        # KEY DIFFERENCE: Context is buried in massive text, parameters may get lost
        prompt = self._create_full_context_prompt(params, full_context)

        # Step 3: Generate with massive context
        agreement = await self._generate_agreement(prompt)

        # Count tokens (approximate)
        context_tokens = len(prompt.split()) * 1.3

        return {
            "agreement_text": agreement,
            "metadata": {
                "total_docs_provided": len(distractor_docs),
                "context_tokens_used": int(context_tokens),
                "approach": "report_system",
                "semantic_filtering_applied": False,
                "compression_applied": False
            }
        }

    def _concatenate_all_context(self, docs: List[str]) -> str:
        """
        Concatenate all documents without filtering or compression.

        This mimics the Report System's approach of providing all
        available context to the LLM.
        """
        if not docs:
            return ""

        # Simple concatenation
        context = "\n\n" + "="*80 + "\n\n"
        context += "AVAILABLE REFERENCE DOCUMENTS:\n\n"

        for i, doc in enumerate(docs):
            context += f"\n\nDocument {i+1}:\n"
            context += "-" * 80 + "\n"
            context += doc
            context += "\n" + "-" * 80 + "\n"

        context += "\n" + "="*80 + "\n\n"

        return context

    def _create_full_context_prompt(self, params: Dict[str, Any], context: str) -> str:
        """
        Create prompt with full context injection.

        KEY DIFFERENCE from TaleStudio:
        - Parameters buried after massive context
        - No clear separation or emphasis
        - LLM must find relevant info in haystack
        """

        # Format parameters (less clearly than TaleStudio)
        params_text = f"""
Task: Generate a credit facility agreement

Borrower: {params['borrower_name']} ({params['borrower_entity_type']}, {params['borrower_state']})
Lender: {params['lender_name']} ({params['lender_state']})
Facility Type: {params['facility_type']}
Principal: ${params['principal_amount']:,.2f}
Interest Rate: {params['interest_rate']}% ({params['interest_rate_type']}, {params['base_rate']} + {params['interest_rate_margin']}%)
Term: {params['term_years']} years ({params['effective_date']} to {params['maturity_date']})
Purpose: {params['purpose']}
Covenants: {', '.join(params['financial_covenants'])}
Collateral: {', '.join(params['security_collateral'])}
Guarantors: {', '.join(params['guarantors']) if params['guarantors'] else 'None'}
Prepayment: {params['prepayment_terms']}
Governing Law: {params['governing_law']}
"""

        # Construct prompt: Context FIRST (Report System pattern)
        # This buries the actual task in massive irrelevant text
        prompt = f"""You are a legal document expert. Below you will find reference documents and parameters for generating a credit facility agreement.

{context}

CREDIT FACILITY PARAMETERS:
{params_text}

Using the above reference documents and parameters, generate a professional credit facility agreement. The agreement should be comprehensive and include all standard sections.

Generate the credit facility agreement now:"""

        return prompt

    async def _generate_agreement(self, prompt: str) -> str:
        """Generate agreement (single attempt, like Report System)."""

        # Report System pattern: Try once with fallback, no extensive validation
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",  # Same model as TaleStudio for fair comparison
                messages=[
                    {"role": "system", "content": "You are an expert legal document drafter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.35,  # Slightly higher than TaleStudio (0.3)
                max_tokens=4096
            )

            agreement = response.choices[0].message.content.strip()
            return agreement

        except Exception as e:
            # Report System pattern: Try fallback with different message format
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "user", "content": f"You are an expert legal document drafter.\n\n{prompt}"}
                    ],
                    temperature=0.35,
                    max_tokens=4096
                )
                return response.choices[0].message.content.strip()
            except Exception as e2:
                print(f"Error in generate_agreement: {e2}")
                return f"ERROR: Failed to generate agreement - {str(e2)}"


async def run_single_test(
    test_id: int,
    params: Dict[str, Any],
    distractor_docs: List[str],
    api_key: str
) -> Dict[str, Any]:
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"Test {test_id}: Report System Approach")
    print(f"{'='*60}")
    print(f"Borrower: {params['borrower_name']}")
    print(f"Amount: ${params['principal_amount']:,.0f}")
    print(f"Distractor docs: {len(distractor_docs)}")

    approach = ReportSystemApproach(api_key)

    result = await approach.generate_credit_facility(params, distractor_docs)

    # Add test metadata
    result["test_id"] = test_id
    result["params"] = params
    result["distractor_count"] = len(distractor_docs)

    print(f"\nResult:")
    print(f"  Agreement length: {len(result['agreement_text'].split())} words")
    print(f"  Total docs provided: {result['metadata']['total_docs_provided']}")
    print(f"  Context tokens: {result['metadata']['context_tokens_used']}")

    return result


async def main():
    """Main test function."""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Import utilities
    from utils.dummy_data import generate_batch
    from utils.edgar_fetcher import EdgarFetcher

    # Generate test parameters
    print("Generating test parameters...")
    params_batch = generate_batch(num_samples=3, start_seed=100)

    # Get distractor documents
    print("Fetching distractor documents (subscription agreements)...")
    fetcher = EdgarFetcher()
    distractor_docs = fetcher.get_sample_subscription_agreements(5)

    # Run tests
    results = []
    for i, params in enumerate(params_batch):
        result = await run_single_test(
            test_id=i,
            params=params,
            distractor_docs=distractor_docs,
            api_key=api_key
        )
        results.append(result)

        # Small delay to avoid rate limits
        await asyncio.sleep(1)

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "report_system_results_sample.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, default=str) + "\n")

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"Total tests: {len(results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
