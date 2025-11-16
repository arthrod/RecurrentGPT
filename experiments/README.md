# Credit Facility Generation Experiment

## Overview

This experiment compares two approaches to long-form document generation:

1. **TaleStudio Approach** - Semantic retrieval + context compression
2. **Report System Approach** - Full context injection

## Hypothesis

The TaleStudio approach (semantic retrieval) will produce higher quality output with fewer hallucinations compared to the Report System approach (full context injection) when dealing with irrelevant distractor documents.

## Experimental Design

### Test Setup

- **Task**: Generate a credit facility agreement
- **Distractor Context**: 5-10 subscription agreements from SEC Edgar (totally irrelevant to credit facilities)
- **Parameters**: Random but realistic credit facility parameters (borrower, lender, amount, terms, etc.)
- **Tests**: 100 iterations per approach
- **LLM**: GPT-4 (same model for both approaches for fair comparison)

### Key Differences

| Aspect | TaleStudio Approach | Report System Approach |
|--------|---------------------|------------------------|
| **Context Handling** | Semantic retrieval (top-2 relevant docs) | Full context injection (all docs) |
| **Filtering** | Yes - filters irrelevant docs | No - includes everything |
| **Compression** | Yes - compresses retrieved context | No - concatenates all |
| **Context Size** | ~2K-4K tokens | ~20K-50K tokens |
| **Parameter Visibility** | Clear, structured prompt | Buried in massive context |
| **Validation** | Multi-attempt with validation | Single attempt with fallback |

### Evaluation Metrics

LLM judge evaluates on:

1. **Overall Score** (0-10): Weighted average of all metrics
2. **Task Adherence** (0-10): Is it actually a credit facility (not subscription agreement)?
3. **Hallucination** (0-10, lower is better): Does it reference distractor documents?
4. **Completeness** (0-10): Does it include all required parameters?
5. **Quality** (0-10): Professional, well-structured, legally sound?
6. **Precision** (0-10): Are values exactly as specified in parameters?

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API (already set in .env file)
# The .env file contains Blackbox AI configuration
# BLACKBOX_API_KEY=sk-q79-se6_7I98XHgodJkHgw
# BLACKBOX_BASE_URL=https://api.blackbox.ai
# BLACKBOX_MODEL=blackboxai/google/gemini-2.5-flash-lite-preview-06-17
```

**Note**: The API configuration is pre-configured in the `.env` file using Blackbox AI's Gemini model.

## Usage

### Quick Test (5 samples)

```bash
cd experiments
python run_full_experiment.py --quick
```

### Full Experiment (100 samples)

```bash
cd experiments
python run_full_experiment.py --num-tests 100
```

### Run Individual Approaches

```bash
# TaleStudio approach only
python test_talestudio_approach.py

# Report System approach only
python test_report_approach.py
```

### Evaluate Existing Results

```bash
python evaluate_results.py
```

## File Structure

```
experiments/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── run_full_experiment.py          # Main experiment runner
├── test_talestudio_approach.py     # TaleStudio implementation
├── test_report_approach.py         # Report System implementation
├── evaluate_results.py             # LLM judge evaluation
├── utils/
│   ├── edgar_fetcher.py            # Fetch subscription agreements
│   └── dummy_data.py               # Generate credit facility parameters
└── results/                        # Output directory
    ├── talestudio_results_*.jsonl
    ├── report_system_results_*.jsonl
    ├── talestudio_evaluations_*.jsonl
    ├── report_system_evaluations_*.jsonl
    └── comparative_analysis_*.json
```

## Expected Results

Based on the analysis in `analysis/04_QUALITY_PRECISION_HALLUCINATIONS.md`, we expect:

### TaleStudio Approach
- **Higher task adherence**: Should ignore subscription agreements, focus on credit facility
- **Lower hallucinations**: Semantic retrieval filters irrelevant docs
- **Higher precision**: Parameters clearly structured in prompt
- **Higher quality**: Focused context enables better output

### Report System Approach
- **Lower task adherence**: May get confused by subscription agreements
- **Higher hallucinations**: All context injected, may reference irrelevant docs
- **Lower precision**: Parameters buried in massive context
- **Lower quality**: Context overload degrades performance

## Sample Output

### Test Case
```json
{
  "borrower_name": "TechVentures Inc.",
  "lender_name": "Capital Trust Bank",
  "principal_amount": 5000000,
  "interest_rate": 7.5,
  "term_years": 5,
  ...
}
```

### TaleStudio Result
```
Retrieved 0 documents (subscription agreements correctly ignored)
Context tokens: ~2,500
Quality: Professional credit facility agreement
Hallucinations: None detected
```

### Report System Result
```
Provided 10 documents (all subscription agreements included)
Context tokens: ~35,000
Quality: May reference subscription terms inappropriately
Hallucinations: Possible references to irrelevant content
```

## Cost Estimation

Using Blackbox AI with Google Gemini 2.5 Flash Lite model:

- **Per test**: ~$0.001-0.01 (significantly cheaper than GPT-4)
- **100 tests per approach**: ~$0.10-2.00
- **Total for full experiment**: ~$0.50-5.00
- **Evaluation (LLM judge)**: ~$0.50-5.00
- **Grand total**: ~$1-10 (much more affordable than GPT-4)

## Analysis

After running the experiment, check:

1. **Comparative Analysis**: `results/comparative_analysis_*.json`
2. **Individual Results**: `results/*_results_*.jsonl`
3. **Detailed Evaluations**: `results/*_evaluations_*.jsonl`

### Key Questions

1. Does TaleStudio approach avoid hallucinations from distractor docs?
2. Does Report System get confused by irrelevant context?
3. Which approach produces more accurate credit facilities?
4. What is the quality difference?
5. How do context sizes compare?

## Citation

This experiment is based on the comparative analysis in:
- `analysis/01_EXECUTIVE_SUMMARY.md`
- `analysis/02_DETAILED_COMPARISON.md`
- `analysis/03_RECOMMENDATIONS.md`
- `analysis/04_QUALITY_PRECISION_HALLUCINATIONS.md`

## License

MIT
