"""
Test TaleStudio approach: Semantic retrieval + focused context.

This script tests whether semantic retrieval can ignore irrelevant
distractor documents (subscription agreements) and focus on the actual
task (generating a credit facility).
"""

import os
import json
import asyncio
from typing import Dict, List, Any
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "torch"])
    from sentence_transformers import SentenceTransformer
    import torch

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Blackbox AI Configuration
BLACKBOX_API_KEY = os.getenv("BLACKBOX_API_KEY")
BLACKBOX_BASE_URL = os.getenv("BLACKBOX_BASE_URL")
BLACKBOX_MODEL = os.getenv("BLACKBOX_MODEL")


class SemanticRetriever:
    """Semantic retrieval for finding relevant context."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def retrieve_relevant(
        self,
        query: str,
        documents: List[str],
        top_k: int = 2,
        threshold: float = 0.3
    ) -> List[str]:
        """
        Retrieve most relevant documents for query using cosine similarity.

        Args:
            query: The query text
            documents: List of document texts
            top_k: Number of top documents to return
            threshold: Minimum similarity threshold

        Returns:
            List of top-k relevant documents
        """
        if not documents:
            return []

        # Encode query and documents
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        doc_embeddings = self.model.encode(documents, convert_to_tensor=True)

        # Calculate cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(1),
            doc_embeddings.unsqueeze(0),
            dim=2
        )[0]

        # Get top-k above threshold
        above_threshold = (similarities >= threshold).nonzero(as_tuple=True)[0]

        if len(above_threshold) == 0:
            # Nothing passes threshold, return empty (key difference from Report System)
            return []

        filtered_sims = similarities[above_threshold]
        top_k_in_filtered = torch.topk(
            filtered_sims,
            k=min(top_k, len(filtered_sims))
        )[1]
        top_k_idx = above_threshold[top_k_in_filtered]

        return [documents[idx] for idx in top_k_idx.tolist()]


class ContextCompressor:
    """Compress context to maintain only key information."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(
            api_key=BLACKBOX_API_KEY,
            base_url=BLACKBOX_BASE_URL
        )

    async def compress(
        self,
        context: str,
        query: str,
        max_words: int = 200
    ) -> str:
        """
        Compress context to max_words while preserving key information.

        Args:
            context: Original context
            query: Query to guide compression
            max_words: Maximum words in compressed output

        Returns:
            Compressed context
        """
        if len(context.split()) <= max_words:
            return context

        prompt = f"""Extract the key information from the following context that is relevant to this query: "{query}"

Context:
{context[:3000]}  # Truncate if too long

Provide a compressed summary in EXACTLY {max_words} words or less that captures:
1. Key facts relevant to the query
2. Important details needed to answer the query
3. Any critical information

Compressed summary ({max_words} words max):"""

        response = await self.client.chat.completions.create(
            model=BLACKBOX_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_words * 2
        )

        return response.choices[0].message.content.strip()


class TaleStudioApproach:
    """
    TaleStudio-style generation: Semantic retrieval + context compression.
    """

    def __init__(self, api_key: str = None):
        self.api_key = BLACKBOX_API_KEY
        self.client = AsyncOpenAI(
            api_key=BLACKBOX_API_KEY,
            base_url=BLACKBOX_BASE_URL
        )
        self.retriever = SemanticRetriever()
        self.compressor = ContextCompressor(BLACKBOX_API_KEY)

    async def generate_credit_facility(
        self,
        params: Dict[str, Any],
        distractor_docs: List[str]
    ) -> Dict[str, Any]:
        """
        Generate credit facility agreement with TaleStudio approach.

        Args:
            params: Credit facility parameters
            distractor_docs: Irrelevant documents (subscription agreements)

        Returns:
            {
                "agreement_text": str,
                "metadata": {
                    "retrieved_docs_count": int,
                    "context_tokens_used": int,
                    "approached": "talestudio"
                }
            }
        """
        # Step 1: Formulate query from parameters
        query = self._params_to_query(params)

        # Step 2: Semantic retrieval - find relevant documents
        # KEY: This should return empty list since subscription agreements
        # are irrelevant to credit facility generation
        retrieved_docs = self.retriever.retrieve_relevant(
            query=query,
            documents=distractor_docs,
            top_k=2,
            threshold=0.4  # Higher threshold to filter noise
        )

        print(f"  Retrieved {len(retrieved_docs)} documents (expected: 0, since docs are irrelevant)")

        # Step 3: Compress retrieved context (if any)
        compressed_context = ""
        if retrieved_docs:
            combined = "\n\n".join(retrieved_docs)
            compressed_context = await self.compressor.compress(
                context=combined,
                query=query,
                max_words=200
            )

        # Step 4: Create focused prompt with ONLY relevant information
        # KEY: Parameters go directly to prompt, not buried in massive context
        focused_prompt = self._create_focused_prompt(params, compressed_context)

        # Step 5: Generate with small, focused context
        agreement = await self._generate_agreement(focused_prompt)

        # Count tokens (approximate)
        context_tokens = len(focused_prompt.split()) * 1.3

        return {
            "agreement_text": agreement,
            "metadata": {
                "retrieved_docs_count": len(retrieved_docs),
                "context_tokens_used": int(context_tokens),
                "approach": "talestudio",
                "semantic_filtering_applied": True,
                "compression_applied": len(retrieved_docs) > 0
            }
        }

    def _params_to_query(self, params: Dict[str, Any]) -> str:
        """Convert parameters to semantic query."""
        return f"""Generate a {params['facility_type']} credit facility agreement for
{params['borrower_name']} borrowing ${params['principal_amount']:,.0f} from
{params['lender_name']} at {params['interest_rate']}% interest for
{params['term_years']} years."""

    def _create_focused_prompt(self, params: Dict[str, Any], context: str) -> str:
        """Create focused prompt with clear structure."""

        # Format parameters clearly
        params_section = f"""
BORROWER INFORMATION:
- Name: {params['borrower_name']}
- Entity Type: {params['borrower_entity_type']}
- State: {params['borrower_state']}

LENDER INFORMATION:
- Name: {params['lender_name']}
- State: {params['lender_state']}

FACILITY TERMS:
- Type: {params['facility_type']}
- Principal Amount: ${params['principal_amount']:,.2f} ({params['principal_amount_words']})
- Interest Rate: {params['interest_rate']}% ({params['interest_rate_type']})
- Base Rate: {params['base_rate']} + {params['interest_rate_margin']}% margin
- Default Rate: {params['default_interest_rate']}%
- Term: {params['term_years']} years
- Effective Date: {params['effective_date']}
- Maturity Date: {params['maturity_date']}
- Purpose: {params['purpose']}

FINANCIAL COVENANTS:
{chr(10).join(f"- {c}" for c in params['financial_covenants'])}

SECURITY/COLLATERAL:
{chr(10).join(f"- {s}" for s in params['security_collateral'])}

GUARANTORS:
{chr(10).join(f"- {g}" for g in params['guarantors']) if params['guarantors'] else "- None"}

PREPAYMENT TERMS:
- {params['prepayment_terms']}

FEES:
- Origination Fee: ${params['fees']['origination_fee']:,.2f}
- Commitment Fee: {params['fees']['commitment_fee']}
- Administrative Fee: ${params['fees']['administrative_fee']:,.2f}
- Legal Fees: {params['fees']['legal_fees']}

GOVERNING LAW:
- {params['governing_law']}
"""

        # Context section (likely empty since docs are irrelevant)
        context_section = ""
        if context:
            context_section = f"\n\nRELEVANT CONTEXT:\n{context}\n"

        # Full prompt with clear instructions
        prompt = f"""You are a legal document expert. Generate a professional credit facility agreement based on the following parameters.

{params_section}{context_section}

INSTRUCTIONS:
1. Create a complete, professional credit facility agreement
2. Include all standard sections: definitions, facility terms, representations and warranties, covenants, events of default, etc.
3. Use formal legal language appropriate for commercial lending
4. Ensure all parameters above are accurately incorporated
5. Length: approximately 3000-4000 words
6. Do NOT include information not specified in the parameters
7. Do NOT reference or use information from any subscription agreements

Generate the credit facility agreement now:"""

        return prompt

    async def _generate_agreement(self, prompt: str) -> str:
        """Generate agreement with retry logic."""
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=BLACKBOX_MODEL,  # Use Blackbox AI Gemini model
                    messages=[
                        {"role": "system", "content": "You are an expert legal document drafter specializing in credit facilities."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Low temperature for consistency
                    max_tokens=4096
                )

                agreement = response.choices[0].message.content.strip()

                # Validation
                if len(agreement.split()) < 500:
                    print(f"  Attempt {attempt + 1}: Agreement too short, retrying...")
                    continue

                if "subscription" in agreement.lower() and "subscription agreement" in agreement.lower():
                    print(f"  Attempt {attempt + 1}: Hallucination detected (subscription agreement mentioned), retrying...")
                    continue

                return agreement

            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise

        return agreement


async def run_single_test(
    test_id: int,
    params: Dict[str, Any],
    distractor_docs: List[str],
    api_key: str
) -> Dict[str, Any]:
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"Test {test_id}: TaleStudio Approach")
    print(f"{'='*60}")
    print(f"Borrower: {params['borrower_name']}")
    print(f"Amount: ${params['principal_amount']:,.0f}")
    print(f"Distractor docs: {len(distractor_docs)}")

    approach = TaleStudioApproach(api_key)

    result = await approach.generate_credit_facility(params, distractor_docs)

    # Add test metadata
    result["test_id"] = test_id
    result["params"] = params
    result["distractor_count"] = len(distractor_docs)

    print(f"\nResult:")
    print(f"  Agreement length: {len(result['agreement_text'].split())} words")
    print(f"  Retrieved docs: {result['metadata']['retrieved_docs_count']}")
    print(f"  Context tokens: {result['metadata']['context_tokens_used']}")

    return result


async def main():
    """Main test function."""
    # Using Blackbox AI (configured at top of file)
    api_key = BLACKBOX_API_KEY

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

    output_file = output_dir / "talestudio_results_sample.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, default=str) + "\n")

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"Total tests: {len(results)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
