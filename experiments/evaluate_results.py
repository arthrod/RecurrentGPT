"""
LLM Judge evaluation script.

Evaluates generated credit facilities on:
1. Task Adherence - Did it generate a credit facility (not subscription agreement)?
2. Hallucination - Does it mention irrelevant distractor content?
3. Completeness - Does it include all required parameters?
4. Quality - Is it professional and well-structured?
5. Precision - How accurately does it use the provided parameters?
"""

import os
import json
import asyncio
from typing import Dict, List, Any
from pathlib import Path
from openai import AsyncOpenAI
import statistics


class LLMJudge:
    """LLM-based judge for evaluating credit facility agreements."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def evaluate(
        self,
        agreement_text: str,
        params: Dict[str, Any],
        distractor_docs: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a generated credit facility agreement.

        Returns scoring across multiple dimensions.
        """
        print("Evaluating agreement...")

        # Run evaluations in parallel
        results = await asyncio.gather(
            self._evaluate_task_adherence(agreement_text),
            self._evaluate_hallucination(agreement_text, distractor_docs),
            self._evaluate_completeness(agreement_text, params),
            self._evaluate_quality(agreement_text),
            self._evaluate_precision(agreement_text, params)
        )

        task_adherence, hallucination, completeness, quality, precision = results

        # Calculate overall score
        overall = statistics.mean([
            task_adherence["score"],
            (10 - hallucination["score"]),  # Invert hallucination (lower is better)
            completeness["score"],
            quality["score"],
            precision["score"]
        ])

        return {
            "overall_score": round(overall, 2),
            "task_adherence": task_adherence,
            "hallucination": hallucination,
            "completeness": completeness,
            "quality": quality,
            "precision": precision
        }

    async def _evaluate_task_adherence(self, agreement: str) -> Dict[str, Any]:
        """
        Evaluate if output is actually a credit facility agreement.

        Score 0-10:
        10 = Perfect credit facility agreement
        0 = Completely wrong document type (e.g., subscription agreement)
        """
        prompt = f"""Evaluate if the following document is a CREDIT FACILITY AGREEMENT (not a subscription agreement, not a partnership agreement, not any other type of document).

Document:
{agreement[:4000]}

Score from 0-10:
- 10: Clearly a credit facility/loan agreement with proper structure
- 7-9: Mostly credit facility but has some issues
- 4-6: Mixed or confused document type
- 1-3: Wrong document type but mentions credit
- 0: Completely wrong document type

Provide your evaluation in JSON format:
{{
    "score": <0-10>,
    "document_type": "<identified document type>",
    "reasoning": "<brief explanation>",
    "issues": ["<list of issues if any>"]
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    async def _evaluate_hallucination(
        self,
        agreement: str,
        distractor_docs: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if output hallucinates information from distractor documents.

        Score 0-10 (higher is WORSE):
        10 = Severe hallucinations, uses distractor content extensively
        0 = No hallucinations, ignores distractor documents appropriately
        """
        # Extract key phrases from distractor docs
        distractor_samples = []
        for doc in distractor_docs[:2]:  # Sample first 2 docs
            lines = doc.split('\n')
            # Get company names, amounts, etc.
            for line in lines[:50]:
                if any(keyword in line for keyword in ['Corporation', 'LLC', 'Inc.', '$', 'subscription', 'Subscriber']):
                    distractor_samples.append(line.strip())

        prompt = f"""Evaluate if the credit facility agreement inappropriately uses or references information from these SUBSCRIPTION AGREEMENTS (which should be IGNORED since the task is to generate a CREDIT FACILITY):

Distractor Document Sample Phrases:
{chr(10).join(distractor_samples[:20])}

Generated Agreement:
{agreement[:4000]}

Check for hallucinations:
1. Does it mention "subscription" or "subscriber" (should not, since it's a credit facility)?
2. Does it reference companies/entities from the subscription agreements?
3. Does it use terms/amounts from subscription agreements instead of the provided parameters?

Score from 0-10 (HIGHER IS WORSE):
- 10: Severe hallucination - extensively uses distractor content
- 7-9: Moderate hallucination - mentions distractor content multiple times
- 4-6: Minor hallucination - occasional inappropriate references
- 1-3: Minimal hallucination - one or two minor mentions
- 0: No hallucination - appropriately ignores all distractor content

Provide evaluation in JSON:
{{
    "score": <0-10>,
    "hallucinated_elements": ["<list of hallucinated content>"],
    "reasoning": "<explanation>",
    "severity": "<none|low|medium|high>"
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    async def _evaluate_completeness(
        self,
        agreement: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate if all required parameters are included in the agreement.

        Score 0-10:
        10 = All parameters accurately included
        0 = Most parameters missing
        """
        required_elements = {
            "Borrower Name": params['borrower_name'],
            "Lender Name": params['lender_name'],
            "Principal Amount": f"${params['principal_amount']:,.0f}",
            "Interest Rate": f"{params['interest_rate']}%",
            "Term": f"{params['term_years']} year",
            "Facility Type": params['facility_type'],
            "Effective Date": params['effective_date'],
            "Maturity Date": params['maturity_date'],
        }

        elements_list = "\n".join([f"- {k}: {v}" for k, v in required_elements.items()])

        prompt = f"""Evaluate if the credit facility agreement includes all required parameters.

Required Parameters:
{elements_list}

Financial Covenants Required:
{chr(10).join(f"- {c}" for c in params['financial_covenants'][:3])}

Security/Collateral Required:
{chr(10).join(f"- {s}" for s in params['security_collateral'][:2])}

Generated Agreement:
{agreement[:4000]}

Check which parameters are correctly included. Score from 0-10:
- 10: All parameters present and accurate
- 7-9: Most parameters present, minor omissions
- 4-6: About half the parameters present
- 1-3: Few parameters present
- 0: Most parameters missing

Provide evaluation in JSON:
{{
    "score": <0-10>,
    "parameters_found": ["<list of correctly included parameters>"],
    "parameters_missing": ["<list of missing parameters>"],
    "parameters_incorrect": ["<list of incorrectly stated parameters>"],
    "reasoning": "<explanation>"
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    async def _evaluate_quality(self, agreement: str) -> Dict[str, Any]:
        """
        Evaluate overall quality and professionalism.

        Score 0-10:
        10 = Professional, well-structured, legally sound
        0 = Poor quality, unprofessional, many errors
        """
        prompt = f"""Evaluate the quality and professionalism of this credit facility agreement.

Agreement:
{agreement[:4000]}

Evaluate on:
1. Structure - Proper sections (definitions, terms, covenants, defaults, etc.)
2. Legal language - Professional, formal, legally appropriate
3. Clarity - Clear and unambiguous terms
4. Completeness - Comprehensive coverage of standard provisions
5. Formatting - Proper organization and readability

Score from 0-10:
- 9-10: Excellent - professional, comprehensive, legally sound
- 7-8: Good - mostly professional with minor issues
- 5-6: Acceptable - adequate but has notable gaps
- 3-4: Poor - significant quality issues
- 0-2: Very poor - unprofessional or incoherent

Provide evaluation in JSON:
{{
    "score": <0-10>,
    "strengths": ["<list of strengths>"],
    "weaknesses": ["<list of weaknesses>"],
    "reasoning": "<explanation>"
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    async def _evaluate_precision(
        self,
        agreement: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate precision - how accurately parameters are used.

        Score 0-10:
        10 = Perfect precision, all values exactly match parameters
        0 = Many values wrong or hallucinated
        """
        prompt = f"""Evaluate the precision of this credit facility agreement - do the specific values match the required parameters?

Required Values:
- Principal: ${params['principal_amount']:,.0f}
- Interest Rate: {params['interest_rate']}%
- Term: {params['term_years']} years
- Borrower: {params['borrower_name']}
- Lender: {params['lender_name']}

Agreement:
{agreement[:4000]}

Check for:
1. Are dollar amounts exact?
2. Are percentages exact?
3. Are names spelled correctly?
4. Are dates consistent?
5. Are any values hallucinated or invented?

Score from 0-10:
- 10: All values exactly match parameters
- 7-9: Minor discrepancies (rounding, formatting)
- 4-6: Several values differ from parameters
- 1-3: Many values incorrect
- 0: Most values hallucinated or wrong

Provide evaluation in JSON:
{{
    "score": <0-10>,
    "exact_matches": ["<list of correctly used values>"],
    "discrepancies": ["<list of mismatches>"],
    "reasoning": "<explanation>"
}}"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)


async def evaluate_results_file(results_file: Path, api_key: str) -> List[Dict]:
    """Evaluate all results in a JSONL file."""
    judge = LLMJudge(api_key)

    evaluations = []

    with open(results_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            result = json.loads(line)

            print(f"\nEvaluating test {result['test_id']}...")

            evaluation = await judge.evaluate(
                agreement_text=result['agreement_text'],
                params=result['params'],
                distractor_docs=[]  # Not needed for most evaluations
            )

            evaluations.append({
                "test_id": result['test_id'],
                "approach": result['metadata']['approach'],
                "evaluation": evaluation,
                "metadata": result['metadata']
            })

            # Rate limiting
            await asyncio.sleep(1)

    return evaluations


async def main():
    """Main evaluation function."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    results_dir = Path("experiments/results")

    # Evaluate both approaches
    for approach_name in ["talestudio", "report_system"]:
        results_file = results_dir / f"{approach_name}_results_sample.jsonl"

        if not results_file.exists():
            print(f"Skipping {approach_name} - results file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {approach_name} approach")
        print(f"{'='*60}")

        evaluations = await evaluate_results_file(results_file, api_key)

        # Save evaluations
        eval_file = results_dir / f"{approach_name}_evaluations.jsonl"
        with open(eval_file, 'w') as f:
            for eval in evaluations:
                f.write(json.dumps(eval, indent=2) + "\n")

        print(f"\nEvaluations saved to {eval_file}")

        # Print summary
        scores = [e['evaluation']['overall_score'] for e in evaluations]
        hallucinations = [e['evaluation']['hallucination']['score'] for e in evaluations]

        print(f"\nSummary for {approach_name}:")
        print(f"  Overall Score: {statistics.mean(scores):.2f} ± {statistics.stdev(scores) if len(scores) > 1 else 0:.2f}")
        print(f"  Hallucination: {statistics.mean(hallucinations):.2f} (lower is better)")
        print(f"  Task Adherence: {statistics.mean([e['evaluation']['task_adherence']['score'] for e in evaluations]):.2f}")
        print(f"  Completeness: {statistics.mean([e['evaluation']['completeness']['score'] for e in evaluations]):.2f}")
        print(f"  Quality: {statistics.mean([e['evaluation']['quality']['score'] for e in evaluations]):.2f}")
        print(f"  Precision: {statistics.mean([e['evaluation']['precision']['score'] for e in evaluations]):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
