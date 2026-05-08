# MCP Server - Loan Default Predictor

This wraps our trained XGBoost model as an MCP tool so Claude Desktop can score loans from a plain text description. You describe the loan in chat and Claude calls `predict_loan_default` behind the scenes.

## Before you start

You need to have run the pipeline at least once so these two files exist:

- `data/processed/preprocessing_pipeline.pkl`
- `models/best_model.pkl`

You also need Claude Desktop installed.

## Setup

```bash
pip install -r requirements.txt
```

## Connect it to Claude Desktop

Open the Claude Desktop config file:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

Add a server entry. Theres a template at [mcp_config_template.json](../../mcp_config_template.json) you can copy in. Replace the path with wherever you cloned the repo:

```json
{
  "mcpServers": {
    "lending-default": {
      "command": "python",
      "args": ["REPLACE_THIS_WITH_YOUR_PROJECT_ROOT/src/mcp/server.py"]
    }
  }
}
```

Restart Claude Desktop. The `lending-default-predictor` connector should show up under the '+' menu.

## The tool: `predict_loan_default`

### Required inputs

| Parameter | Type | Description | Example |
|---|---|---|---|
| `loan_amnt` | float | Loan amount in USD | `15000` |
| `term` | int | Loan term, 36 or 60 months | `36` |
| `int_rate` | float | Interest rate (%) | `12.5` |
| `sub_grade` | str | A1 (lowest risk) to G5 (highest) | `"B3"` |
| `annual_inc` | float | Annual income in USD | `65000` |
| `dti` | float | Debt-to-income ratio (%) | `18.5` |
| `fico_score` | int | FICO score, 300-850 | `710` |
| `home_ownership` | str | RENT / OWN / MORTGAGE / OTHER | `"RENT"` |
| `purpose` | str | See list below | `"debt_consolidation"` |

Valid `purpose` values: `debt_consolidation`, `credit_card`, `home_improvement`, `medical`, `small_business`, `car`, `vacation`, `moving`, `house`, `major_purchase`, `other`, `wedding`, `renewable_energy`, `educational`

### Optional inputs (default to dataset medians if you skip them)

| Parameter | Default | Description |
|---|---|---|
| `open_acc` | `10` | Open credit lines |
| `revol_util` | `30.0` | Revolving utilization % |
| `total_acc` | `20` | Total credit lines ever |
| `delinq_2yrs` | `0` | Delinquencies in past 2 years |
| `pub_rec` | `0` | Public derogatory records |
| `inq_last_6mths` | `0` | Credit inquiries in last 6 months |
| `verification_status` | `"Not Verified"` | Not Verified / Verified / Source Verified |
| `earliest_cr_line` | `"2010-01"` | Oldest credit line date (YYYY-MM) |

### Output

```json
{
  "default_probability": 0.2847,
  "prediction": "LOW RISK",
  "risk_level": 2,
  "risk_description": "Low risk (28.5%), Recommendation: Approve",
  "model": "XGBClassifier"
}
```

Risk tiers: 1 = Very Low (<15%, Approve), 2 = Low (15-25%, Approve), 3 = Moderate (25-35%, Review), 4 = High (35-50%, Decline), 5 = Very High (>=50%, Decline).

If any input fails validation the tool returns `{"error": "..."}` and never runs the model.

## Example prompts

> "Predict default risk for a $12,000, 36-month loan at 11.5% interest. Sub-grade B2, $58,000 income, DTI 15%, FICO 720, renting, debt consolidation."

> "Compare two loans: (1) $5,000 at 7%, A1 grade, FICO 780, $90K income vs (2) $25,000 at 22%, E4 grade, FICO 620, $35K income. Which is riskier?"

## What happens under the hood

When Claude calls the tool, `server.py`:

1. validates the inputs
2. builds a DataFrame with the user inputs plus median defaults for secondary columns
3. runs `pipeline.transform()`, the same preprocessing we used in training (drop correlated features, date extraction, feature engineering, outlier capping, missing indicators, FRED unemployment join, rare-category merge, scaling, encoding)
4. runs `model.predict_proba()` and returns the probability + risk tier

Both `preprocessing_pipeline.pkl` and `best_model.pkl` are loaded once at startup, not per-call.
