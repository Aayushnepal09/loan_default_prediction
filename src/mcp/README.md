# MCP Server — Loan Default Predictor

This exposes our trained XGBoost model as an MCP tool so Claude Desktop can predict loan default risk from a plain text description. You just describe the loan in chat and Claude calls `predict_loan_default` behind the scenes.

## Before you start

Make sure you've run the full pipeline at least once — you need these two files to exist:
- `data/processed/preprocessing_pipeline.pkl`
- `models/best_model.pkl`

Also needs Claude Desktop installed.

## Setup

```bash
pip install -r requirements.txt
```

## Connecting to Claude Desktop

**1. Open the config file:**
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

**2. Add the server entry:**

Use the template at [mcp_config_template.json](../../mcp_config_template.json) and replace the path with wherever you cloned the repo:

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

**3. Restart Claude Desktop.** The `lending-default-predictor` connector should show up (click '+' to verify).

## Tool: `predict_loan_default`

### Required inputs

| Parameter | Type | Description | Example |
|---|---|---|---|
| `loan_amnt` | float | Loan amount in USD | `15000` |
| `term` | int | Loan term — 36 or 60 months only | `36` |
| `int_rate` | float | Interest rate (%) | `12.5` |
| `sub_grade` | str | A1 (lowest risk) to G5 (highest) | `"B3"` |
| `annual_inc` | float | Annual income in USD | `65000` |
| `dti` | float | Debt-to-income ratio (%) | `18.5` |
| `fico_score` | int | FICO score, 300–850 | `710` |
| `home_ownership` | str | RENT / OWN / MORTGAGE / OTHER | `"RENT"` |
| `purpose` | str | See list below | `"debt_consolidation"` |

Valid `purpose` values: `debt_consolidation`, `credit_card`, `home_improvement`, `medical`, `small_business`, `car`, `vacation`, `moving`, `house`, `major_purchase`, `other`, `wedding`, `renewable_energy`, `educational`

### Optional inputs (defaults are dataset medians)

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
  "risk_description": "Low risk (28.5%) — Recommendation: Approve",
  "model": "XGBClassifier"
}
```

Risk levels: 1 = Very Low (<15%, Approve), 2 = Low (15–25%, Approve), 3 = Moderate (25–35%, Review), 4 = High (35–50%, Decline), 5 = Very High (≥50%, Decline)

If any input fails validation the tool returns `{"error": "..."}` without running the model.

## Example prompts

> "Predict default risk for a $12,000, 36-month loan at 11.5% interest. Sub-grade B2, $58,000 income, DTI 15%, FICO 720, renting, debt consolidation."

> "Compare two loans: (1) $5,000 at 7%, A1 grade, FICO 780, $90K income vs (2) $25,000 at 22%, E4 grade, FICO 620, $35K income. Which is riskier?"

## How it works

```
Claude Desktop
     | MCP (stdio)
     v
server.py
  - loads preprocessing_pipeline.pkl and best_model.pkl on startup
  - predict_loan_default():
      1. validates inputs
      2. builds a DataFrame (user inputs + median defaults for secondary columns)
      3. runs pipeline.transform() — same steps as training
      4. runs model.predict_proba() and returns the result
```

The pipeline applies the same transforms used during training: drop correlated features, date extraction, feature engineering, outlier capping, missing indicators, macro join (FRED unemployment), rare category merging, scaling and encoding.
