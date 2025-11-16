"""
Fetch subscription agreements from SEC Edgar for testing.
"""

import requests
import json
import time
import random
from typing import List, Dict
from pathlib import Path
import re


class EdgarFetcher:
    """Fetch documents from SEC Edgar database."""

    def __init__(self, cache_dir: str = "./cache/edgar"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Research Experiment contact@example.com',
            'Accept-Encoding': 'gzip, deflate',
        }

    def search_subscription_agreements(self, max_results: int = 100) -> List[Dict]:
        """
        Search for subscription agreements on Edgar.

        Returns list of filing metadata.
        """
        # Search for 8-K filings which often contain subscription agreements
        # Form types that commonly have subscription agreements as exhibits
        form_types = ['8-K', 'S-1', 'S-3', 'SC 13D']

        results = []

        # Use Edgar full-text search API
        for form_type in form_types:
            if len(results) >= max_results:
                break

            search_url = f"https://efts.sec.gov/LATEST/search-index"
            params = {
                'q': 'subscription agreement',
                'forms': form_type,
                'startdt': '2020-01-01',
                'enddt': '2024-12-31'
            }

            try:
                response = requests.get(search_url, params=params, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    if 'hits' in data and 'hits' in data['hits']:
                        for hit in data['hits']['hits'][:20]:
                            if len(results) >= max_results:
                                break
                            results.append({
                                'cik': hit['_source'].get('ciks', [''])[0],
                                'company': hit['_source'].get('display_names', [''])[0],
                                'form_type': hit['_source'].get('form'),
                                'filing_date': hit['_source'].get('file_date'),
                                'accession': hit['_source'].get('accession_number'),
                            })
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"Error searching {form_type}: {e}")
                continue

        return results

    def get_sample_subscription_agreements(self, num_samples: int = 10) -> List[str]:
        """
        Get sample subscription agreements.

        For demo purposes, we'll use pre-defined CIKs that commonly file these.
        In production, would fetch from Edgar API.
        """
        # Sample companies known to file subscription agreements
        sample_ciks = [
            '0001318605',  # Tesla
            '0001018724',  # Amazon
            '0001652044',  # Alphabet
            '0001326801',  # Meta
            '0000789019',  # Microsoft
            '0001067983',  # Berkshire Hathaway
            '0000320193',  # Apple
            '0001045810',  # Nvidia
            '0001559720',  # Moderna
            '0001513845',  # Coinbase
        ]

        # For this experiment, we'll create realistic subscription agreement text
        # In production, would fetch actual filings
        templates = self._get_subscription_agreement_templates()

        agreements = []
        for i in range(min(num_samples, len(templates))):
            agreements.append(templates[i])

        return agreements

    def _get_subscription_agreement_templates(self) -> List[str]:
        """Generate realistic subscription agreement text."""

        templates = [
            """
SUBSCRIPTION AGREEMENT

THIS SUBSCRIPTION AGREEMENT (this "Agreement") is made as of December 15, 2023, by and between ACME Technology Corporation, a Delaware corporation (the "Company"), and the undersigned subscriber (the "Subscriber").

RECITALS

WHEREAS, the Company is conducting a private placement offering (the "Offering") of its Series B Preferred Stock (the "Preferred Stock") pursuant to that certain Private Placement Memorandum dated November 1, 2023;

WHEREAS, the Subscriber desires to subscribe for and purchase shares of Preferred Stock from the Company on the terms and conditions set forth herein;

NOW, THEREFORE, in consideration of the mutual covenants and agreements set forth herein and for other good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged, the parties agree as follows:

ARTICLE I
SUBSCRIPTION

1.1 Subscription. Subject to the terms and conditions set forth herein, the Subscriber hereby irrevocably subscribes for and agrees to purchase from the Company 500,000 shares of Preferred Stock at a purchase price of $10.00 per share, for an aggregate purchase price of $5,000,000 (the "Purchase Price").

1.2 Closing. The closing of the purchase and sale of the Preferred Stock (the "Closing") shall occur on December 31, 2023, or such other date as mutually agreed by the parties (the "Closing Date").

1.3 Payment. At the Closing, the Subscriber shall pay the Purchase Price by wire transfer of immediately available funds to an account designated by the Company.

ARTICLE II
REPRESENTATIONS AND WARRANTIES OF THE SUBSCRIBER

The Subscriber hereby represents and warrants to the Company as follows:

2.1 Authorization. The Subscriber has full power and authority to enter into this Agreement and to consummate the transactions contemplated hereby.

2.2 Accredited Investor Status. The Subscriber is an "accredited investor" as defined in Rule 501(a) of Regulation D promulgated under the Securities Act of 1933, as amended.

2.3 Investment Intent. The Subscriber is acquiring the Preferred Stock for investment purposes only and not with a view to, or for resale in connection with, any distribution thereof.

2.4 Access to Information. The Subscriber has had the opportunity to ask questions and receive answers from the Company regarding the terms and conditions of the Offering and the business, properties, prospects, and financial condition of the Company.

2.5 Economic Risk. The Subscriber is able to bear the economic risk of the investment in the Preferred Stock, including a complete loss of the Subscriber's investment.

ARTICLE III
REPRESENTATIONS AND WARRANTIES OF THE COMPANY

The Company hereby represents and warrants to the Subscriber as follows:

3.1 Organization and Standing. The Company is a corporation duly organized, validly existing, and in good standing under the laws of the State of Delaware.

3.2 Authorization. The Company has full corporate power and authority to enter into this Agreement and to issue and sell the Preferred Stock to the Subscriber.

3.3 Valid Issuance. The Preferred Stock, when issued and delivered to the Subscriber against payment therefor as provided herein, will be duly and validly issued, fully paid, and nonassessable.

3.4 No Conflicts. The execution, delivery, and performance of this Agreement by the Company do not conflict with or result in a breach of any agreement or instrument to which the Company is a party.

ARTICLE IV
CONDITIONS TO CLOSING

4.1 Conditions to Subscriber's Obligations. The Subscriber's obligation to purchase the Preferred Stock is subject to the satisfaction of the following conditions:

(a) All representations and warranties of the Company contained herein shall be true and correct as of the Closing Date;
(b) The Company shall have performed all obligations required to be performed by it under this Agreement on or prior to the Closing Date;
(c) No action shall have been instituted or threatened before any court or governmental agency to restrain or prohibit the transactions contemplated by this Agreement.

4.2 Conditions to Company's Obligations. The Company's obligation to sell the Preferred Stock is subject to the satisfaction of the following conditions:

(a) All representations and warranties of the Subscriber contained herein shall be true and correct as of the Closing Date;
(b) The Subscriber shall have performed all obligations required to be performed by it under this Agreement on or prior to the Closing Date;
(c) The Company shall have received the Purchase Price in immediately available funds.

ARTICLE V
MISCELLANEOUS

5.1 Governing Law. This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflicts of law principles.

5.2 Entire Agreement. This Agreement constitutes the entire agreement between the parties with respect to the subject matter hereof and supersedes all prior agreements and understandings.

5.3 Amendments. This Agreement may not be amended except by a written instrument signed by both parties.

5.4 Counterparts. This Agreement may be executed in counterparts, each of which shall be deemed an original.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

ACME TECHNOLOGY CORPORATION

By: _______________________
Name: Jane Smith
Title: Chief Executive Officer

SUBSCRIBER

By: _______________________
Name: John Investor
Title: Managing Partner
            """,

            """
SUBSCRIPTION AGREEMENT FOR UNITS

This SUBSCRIPTION AGREEMENT (this "Agreement"), dated as of September 20, 2023, is entered into by and between BLOCKCHAIN VENTURES LLC, a Delaware limited liability company (the "Company"), and the undersigned (the "Investor").

BACKGROUND

The Company is offering for sale up to $50,000,000 in aggregate of units (the "Units"), each Unit consisting of (i) one Class A membership interest and (ii) one warrant to purchase one additional Class A membership interest at an exercise price of $15.00 per interest. The Company is offering the Units in a private placement transaction (the "Offering") pursuant to the Confidential Private Placement Memorandum dated September 1, 2023 (the "Memorandum").

AGREEMENT

1. SUBSCRIPTION

1.1 Purchase of Units. Subject to the terms and conditions hereof, the Investor hereby subscribes for and agrees to purchase 10,000 Units at a purchase price of $20.00 per Unit, for an aggregate subscription amount of $200,000 (the "Subscription Amount").

1.2 Payment Terms. The Investor shall pay the Subscription Amount as follows:
    (a) Initial Payment: $50,000 due upon execution of this Agreement
    (b) Second Payment: $75,000 due on October 15, 2023
    (c) Final Payment: $75,000 due on November 15, 2023

1.3 Acceptance. This subscription is subject to acceptance by the Company. The Company reserves the right to reject this subscription in whole or in part for any reason.

2. INVESTOR REPRESENTATIONS

The Investor represents, warrants, and covenants to the Company as follows:

2.1 Accredited Investor. The Investor is an "accredited investor" as defined in Rule 501 of Regulation D under the Securities Act of 1933, as amended (the "Securities Act"), by reason of satisfying one or more of the following criteria: [Check applicable box]

□ Individual with net worth exceeding $1,000,000 (excluding primary residence)
□ Individual with income exceeding $200,000 in each of the two most recent years
□ Entity with total assets exceeding $5,000,000
□ Trust with total assets exceeding $5,000,000

2.2 Investment Experience. The Investor has such knowledge and experience in financial and business matters that the Investor is capable of evaluating the merits and risks of the investment in the Units.

2.3 Investment Risk. The Investor acknowledges that the investment in the Units involves substantial risk and that the Investor can afford to bear the economic risk of the investment for an indefinite period of time.

2.4 No Registration. The Investor understands that the Units have not been registered under the Securities Act or any state securities laws and are being offered in reliance on exemptions from the registration requirements thereof.

2.5 Restricted Securities. The Investor understands that the Units are "restricted securities" and may not be sold, transferred, or otherwise disposed of without registration under the Securities Act or an exemption therefrom.

3. COMPANY REPRESENTATIONS

The Company represents and warrants to the Investor as follows:

3.1 Organization. The Company is duly organized, validly existing, and in good standing under the laws of Delaware.

3.2 Authorization. The execution and delivery of this Agreement and the issuance of the Units have been duly authorized by all necessary company action.

3.3 Offering Materials. The information contained in the Memorandum is accurate and complete in all material respects as of the date hereof.

4. CLOSING

4.1 Closing Date. The closing of the purchase and sale of the Units (the "Closing") shall take place within five business days after the Company's acceptance of this subscription (the "Closing Date").

4.2 Deliveries at Closing. At the Closing:
    (a) The Investor shall deliver the Subscription Amount to the Company; and
    (b) The Company shall deliver to the Investor evidence of the Investor's ownership of the Units.

5. LOCK-UP PERIOD

The Investor agrees not to sell, transfer, or otherwise dispose of any Units for a period of twelve (12) months from the Closing Date without the prior written consent of the Company.

6. CONFIDENTIALITY

The Investor agrees to keep confidential all non-public information regarding the Company and the Offering and not to disclose such information to any third party without the Company's prior written consent.

7. GOVERNING LAW AND DISPUTE RESOLUTION

7.1 Governing Law. This Agreement shall be governed by the laws of the State of Delaware without regard to conflicts of law principles.

7.2 Arbitration. Any dispute arising out of or relating to this Agreement shall be resolved by binding arbitration in accordance with the rules of the American Arbitration Association.

8. GENERAL PROVISIONS

8.1 Notices. All notices hereunder shall be in writing and delivered by email, certified mail, or overnight courier.

8.2 Assignment. This Agreement may not be assigned by either party without the prior written consent of the other party.

8.3 Severability. If any provision of this Agreement is held invalid or unenforceable, the remaining provisions shall continue in full force and effect.

8.4 Counterparts. This Agreement may be executed in counterparts, each of which shall constitute an original.

[SIGNATURE PAGE FOLLOWS]

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

BLOCKCHAIN VENTURES LLC

By: _______________________
Name: Michael Chen
Title: Managing Director

INVESTOR

Signature: _______________________
Name: Sarah Anderson
Date: _______________________
            """
        ]

        # Add more variations
        for i in range(3, 11):
            # Generate variations with different companies, amounts, dates
            templates.append(self._generate_variation(templates[i % 2], i))

        return templates

    def _generate_variation(self, base_template: str, seed: int) -> str:
        """Generate variation of subscription agreement."""
        random.seed(seed)

        # Replace company names, amounts, dates
        companies = ["TechCorp", "InnovateLabs", "FutureFund", "GrowthCapital",
                     "VentureX", "AlphaPartners", "BetaInvestments", "GammaHoldings"]
        amounts = ["$1,000,000", "$2,500,000", "$5,000,000", "$10,000,000",
                   "$15,000,000", "$25,000,000"]

        variation = base_template
        variation = variation.replace("ACME Technology Corporation", random.choice(companies))
        variation = variation.replace("BLOCKCHAIN VENTURES LLC", random.choice(companies))
        variation = variation.replace("$5,000,000", random.choice(amounts))
        variation = variation.replace("$200,000", random.choice(amounts))

        return variation


if __name__ == "__main__":
    # Test the fetcher
    fetcher = EdgarFetcher()
    agreements = fetcher.get_sample_subscription_agreements(5)
    print(f"Fetched {len(agreements)} subscription agreements")
    print(f"\nFirst agreement preview (first 500 chars):")
    print(agreements[0][:500])
