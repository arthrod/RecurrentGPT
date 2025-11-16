"""
Generate dummy data for credit facility agreements.
"""

import random
from typing import Dict, List
from datetime import datetime, timedelta


class CreditFacilityDataGenerator:
    """Generate realistic but random credit facility parameters."""

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

    def generate_credit_facility_params(self) -> Dict:
        """
        Generate parameters for a credit facility agreement.

        Returns a dict with all necessary parameters.
        """
        borrower = self._random_borrower()
        lender = self._random_lender()
        facility_type = random.choice(['Revolving', 'Term Loan', 'Bridge Loan', 'Letter of Credit'])
        amount = self._random_amount()
        interest_rate = self._random_interest_rate()
        term_years = random.choice([1, 2, 3, 5, 7, 10])
        purpose = self._random_purpose()
        covenants = self._random_covenants()
        security = self._random_security()

        effective_date = datetime.now()
        maturity_date = effective_date + timedelta(days=365 * term_years)

        return {
            "borrower_name": borrower["name"],
            "borrower_state": borrower["state"],
            "borrower_entity_type": borrower["entity_type"],
            "lender_name": lender["name"],
            "lender_state": lender["state"],
            "facility_type": facility_type,
            "principal_amount": amount,
            "principal_amount_words": self._amount_to_words(amount),
            "currency": "USD",
            "interest_rate_type": random.choice(["Fixed", "Variable"]),
            "interest_rate": interest_rate,
            "base_rate": random.choice(["SOFR", "Prime Rate", "LIBOR (Legacy)"]),
            "interest_rate_margin": round(random.uniform(1.0, 4.5), 2),
            "effective_date": effective_date.strftime("%B %d, %Y"),
            "maturity_date": maturity_date.strftime("%B %d, %Y"),
            "term_years": term_years,
            "purpose": purpose,
            "financial_covenants": covenants,
            "security_collateral": security,
            "guarantors": self._random_guarantors(),
            "prepayment_terms": self._random_prepayment(),
            "default_interest_rate": round(interest_rate + 2.0, 2),
            "fees": self._random_fees(amount),
            "governing_law": random.choice(["New York", "Delaware", "California"]),
        }

    def _random_borrower(self) -> Dict:
        """Generate random borrower details."""
        companies = [
            "Acme Manufacturing Corp.",
            "TechVentures Inc.",
            "GreenEnergy Solutions LLC",
            "RetailMax Holdings",
            "BioPharm Innovations Ltd.",
            "DataCloud Systems Inc.",
            "AutoParts Distribution Co.",
            "UrbanDevelopment Partners",
            "AgriTech Farms LLC",
            "LogisticsHub Corporation"
        ]

        states = ["Delaware", "California", "New York", "Texas", "Nevada"]
        entity_types = ["Corporation", "Limited Liability Company", "Limited Partnership"]

        return {
            "name": random.choice(companies),
            "state": random.choice(states),
            "entity_type": random.choice(entity_types)
        }

    def _random_lender(self) -> Dict:
        """Generate random lender details."""
        banks = [
            "First National Bank",
            "Capital Trust Bank",
            "Metropolitan Financial Group",
            "Global Commerce Bank",
            "United Business Lenders",
            "Atlantic Capital Bank",
            "Pacific Lending Corporation",
            "Midwest Regional Bank",
            "Enterprise Finance Company",
            "Commercial Credit Partners"
        ]

        states = ["New York", "California", "Delaware", "Massachusetts", "Illinois"]

        return {
            "name": random.choice(banks),
            "state": random.choice(states)
        }

    def _random_amount(self) -> float:
        """Generate random principal amount."""
        amounts = [
            1_000_000, 2_500_000, 5_000_000, 7_500_000, 10_000_000,
            15_000_000, 20_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000
        ]
        return random.choice(amounts)

    def _random_interest_rate(self) -> float:
        """Generate random interest rate."""
        return round(random.uniform(4.5, 12.0), 2)

    def _random_purpose(self) -> str:
        """Generate random loan purpose."""
        purposes = [
            "Working capital and general corporate purposes",
            "Acquisition financing and business expansion",
            "Refinancing of existing indebtedness",
            "Capital expenditures and equipment purchases",
            "Real estate development and construction",
            "Inventory financing and supply chain management",
            "Research and development activities",
            "Marketing and business development initiatives"
        ]
        return random.choice(purposes)

    def _random_covenants(self) -> List[str]:
        """Generate random financial covenants."""
        all_covenants = [
            "Minimum debt service coverage ratio of 1.25:1",
            "Maximum total leverage ratio of 3.50:1",
            "Minimum fixed charge coverage ratio of 1.15:1",
            "Maximum senior secured leverage ratio of 2.50:1",
            "Minimum liquidity of $5,000,000",
            "Minimum tangible net worth of $10,000,000",
            "Maximum capital expenditures of $2,000,000 per year",
            "Minimum interest coverage ratio of 3.00:1"
        ]
        return random.sample(all_covenants, k=random.randint(3, 5))

    def _random_security(self) -> List[str]:
        """Generate random security/collateral."""
        all_security = [
            "First priority security interest in all accounts receivable",
            "First priority security interest in all inventory",
            "First priority security interest in all equipment and machinery",
            "Pledge of 100% of equity interests in subsidiaries",
            "First mortgage on real property located at borrower's facilities",
            "Assignment of intellectual property rights",
            "Cash collateral account with minimum balance requirement",
            "Personal guarantees from principal shareholders"
        ]
        return random.sample(all_security, k=random.randint(2, 4))

    def _random_guarantors(self) -> List[str]:
        """Generate random guarantors."""
        if random.random() < 0.6:  # 60% chance of having guarantors
            names = [
                "Acme Holdings LLC",
                "TechVentures Parent Corp.",
                "John Smith (CEO)",
                "Jane Doe (Principal Shareholder)",
                "Operating Subsidiary #1",
                "Operating Subsidiary #2"
            ]
            return random.sample(names, k=random.randint(1, 3))
        return []

    def _random_prepayment(self) -> str:
        """Generate random prepayment terms."""
        terms = [
            "Optional prepayment permitted at any time without premium or penalty",
            "Optional prepayment permitted with 30 days' prior notice and 1% prepayment fee",
            "Optional prepayment permitted after 12 months without penalty",
            "Mandatory prepayment required from excess cash flow (50% sweep)",
            "Mandatory prepayment required upon asset sales exceeding $1,000,000"
        ]
        return random.choice(terms)

    def _random_fees(self, principal: float) -> Dict:
        """Generate random fees."""
        return {
            "origination_fee": round(principal * random.uniform(0.005, 0.02), 2),
            "commitment_fee": f"{round(random.uniform(0.25, 0.75), 2)}% per annum on unused portion",
            "administrative_fee": round(random.uniform(5000, 25000), 2),
            "legal_fees": "Actual costs incurred, estimated at $50,000-$100,000"
        }

    def _amount_to_words(self, amount: float) -> str:
        """Convert amount to words (simplified)."""
        millions = amount / 1_000_000
        if millions == int(millions):
            return f"{int(millions):,} Million Dollars"
        else:
            return f"{millions:,.1f} Million Dollars"


def generate_batch(num_samples: int, start_seed: int = 0) -> List[Dict]:
    """Generate batch of credit facility parameters."""
    batch = []
    for i in range(num_samples):
        generator = CreditFacilityDataGenerator(seed=start_seed + i)
        batch.append(generator.generate_credit_facility_params())
    return batch


if __name__ == "__main__":
    # Test the generator
    generator = CreditFacilityDataGenerator(seed=42)
    params = generator.generate_credit_facility_params()

    print("Sample Credit Facility Parameters:")
    print(f"Borrower: {params['borrower_name']}")
    print(f"Lender: {params['lender_name']}")
    print(f"Facility Type: {params['facility_type']}")
    print(f"Amount: ${params['principal_amount']:,.2f} ({params['principal_amount_words']})")
    print(f"Interest Rate: {params['interest_rate']}%")
    print(f"Term: {params['term_years']} years")
    print(f"Purpose: {params['purpose']}")
    print(f"\nFinancial Covenants:")
    for covenant in params['financial_covenants']:
        print(f"  - {covenant}")
