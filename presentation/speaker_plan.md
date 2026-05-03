# Phase 4 Speaker Plan

**Course:** EAS 587 — Spring 2026
**Team:** 404 Team Not Found  ·  Aayush Nepal · Junwei Zhang · Lusi Zhang
**Format:** 15-minute live presentation driven entirely from the Streamlit app
(`streamlit run src/app/streamlit_app.py`). The 9 tabs are the slides.

This document is for rehearsal. Each section lists the owner, a hard time
budget, and 3 talking points + the transition phrase to the next person.
Total speaking time is 15:00. Run a stopwatch in dry runs and aim to come in
30 seconds under, not over.

---

## Time budget at a glance

| #   | Tab                      | Owner   | Time   | Cumulative |
| --- | ------------------------ | ------- | ------ | ---------- |
| 1   | Welcome                  | Lusi    | 1:00   | 1:00       |
| 2   | Phase 1: Data            | Junwei  | 1:30   | 2:30       |
| 3   | Phase 2: Pipeline        | Lusi    | 1:30   | 4:00       |
| 4   | Phase 2: Models          | Lusi    | 2:30   | 6:30       |
| 5   | Phase 3: Spark + Macro   | Aayush  | 2:00   | 8:30       |
| 6   | Phase 3: MCP             | Aayush  | 1:30   | 10:00      |
| 7   | Predict a loan (demo)    | Aayush  | 2:30   | 12:30      |
| 8   | Insights + Next          | Junwei  | 1:30   | 14:00      |
| 9   | Q&A                      | All 3   | 1:00   | 15:00      |

Solo airtime totals: **Aayush 6:00**, **Lusi 5:00**, **Junwei 3:00** + shared
Q&A. Teammates not currently presenting should stand at the laptop side, not
block the screen.

---

## Tab 1 — Welcome (1:00, Lusi)

**Goal:** anchor the audience on the question and the team in 60 seconds.

1. "We're 404 Team Not Found. Lending Club is a peer-to-peer lender. Our use
   case is simple — given a loan application, what is the probability the
   borrower will charge off?"
2. "Same use case, four implementations: pandas → scikit-learn → Spark + MCP
   → this Streamlit app." Point at the four-stat strip: 314k test loans,
   21% default rate, 0.726 test AUC-ROC, 110 features.
3. "Phases 1 and 2 received perfect scores. We retained the time-based split,
   leakage discipline, and the 8-stage pipeline, and built on top."

**Transition:** "Junwei is going to start us off with how the data was
prepared — because the choices there are what made everything else
possible."

---

## Tab 2 — Phase 1: Data (1:30, Junwei)

**Goal:** sell the leakage discipline. It is our most defensible decision.

1. **Filter:** "We kept 2014–2017 only. Pre-2014 is too sparse, post-2017
   loans haven't matured. Roughly 1.36 million completed loans."
2. **Time-based split** (point at the 3-bar chart): "Train + val from
   2014–2016 by issue date; full 2017 held out as test. 314,212 loans the
   model has never seen." Point at the donut: "4-to-1 class imbalance,
   intentional. We do not resample."
3. **Leakage discipline** (point at the table): "We dropped 21 post-loan
   columns — total payments, recoveries, last FICO, settlement fields. Most
   public Lending Club notebooks accidentally train on these and report
   inflated metrics. Ours don't."

**Transition:** "Lusi takes us into the modeling work — pipeline first."

---

## Tab 3 — Phase 2: Pipeline (1:30, Lusi)

**Goal:** show the 8-stage preprocessing pipeline as one coherent story.

1. **The diagram** (point left to right): "Eight custom scikit-learn
   transformers, fit on train only, applied identically to val and test.
   Drop correlated columns, extract dates, engineer features, cap outliers,
   add missingness flags, **join FRED unemployment**, merge rare categories,
   final routing."
2. **Engineered features:** "loan-to-income, monthly-payment-to-income from
   the actual PMT formula, delinquency rate. These help the linear model
   especially."
3. **Why this matters:** "Same fit-on-train-only pipeline runs in the MCP
   server in production. One artifact, two consumers."

**Transition:** "Now the model bake-off — we tried four algorithms."

---

## Tab 4 — Phase 2: Models (2:30, Lusi)

**Goal:** the technical climax. Bake-off, holdout, threshold tuner.

1. **Bake-off chart** (point): "Logistic Regression baseline, then three
   tuned tree models — Optuna, 50 trials each, MLflow tracked. XGBoost won at
   0.7238 val AUC-ROC, narrowly above HistGradientBoosting; XGBoost is 5x
   faster to fit and exposes native SHAP contributions."
2. **Holdout test row** (point at 4 metric callouts): "0.726 AUC-ROC on
   314,212 unseen 2017 loans. AUC-PR 0.41 against a 21% prior — almost 2x
   lift. KS 0.328. At Youden's J threshold we catch 67% of defaults."
3. **Threshold tuner** (live demo): "Move the slider. Watch precision and
   recall trade off live, confusion matrix updates against the full test set,
   the operating point on the ROC moves." Slide from 0.50 → 0.30 → 0.70 to
   show the curve. "A lender can pick the operating point matched to their
   cost ratio. This is what makes the model deployable, not just trained."

**Transition:** "Aayush will take Phase 3 — Spark, Delta Lake, and the FRED
macro layer."

---

## Tab 5 — Phase 3: Spark + Macro (2:00, Aayush)

**Goal:** show that we scaled and added the rubric-rewarded secondary data.

1. **Medallion architecture** (point at 3 cards): "Bronze, Silver, Gold —
   Kaggle archive landed as Delta, cleaned, then time-split with stratified
   train/val. All Spark, all Delta tables. We rebuilt Phase 1 + 2 on
   Databricks Free Edition."
2. **FRED secondary data:** "Eight series — unemployment, Fed Funds, CPI,
   plus state unemployment for the 5 highest-volume states. Plus 4 engineered
   features: real interest rate, rate spread, state unemployment, CPI YoY."
3. **Three insight charts** (scroll down, point at each): "Default rate vs
   national unemployment at issuance, sub-grade by Fed Funds regime — the
   gap widens in tighter monetary policy, and state unemployment differential
   vs default rate." Then point at the honest finding: **"Adding 7 macro
   features beyond the production UNRATE did NOT improve LR AUC. We report
   that honestly. The macro work earns its place through the insights, not
   model lift."**

**Transition:** "And the same model is also deployed conversationally."

---

## Tab 6 — Phase 3: MCP (1:30, Aayush)

**Goal:** introduce the differentiator. Most teams won't have an MCP server.

1. **What MCP is** (point at the architecture block): "Model Context Protocol
   — Anthropic's standard for letting an LLM call your tools. Our FastMCP
   server loads the same pickle files the Streamlit app does, exposes
   `predict_loan_default` over stdio, validates inputs, runs the same
   transform pipeline."
2. **Why it matters:** "The model becomes reachable two ways — the
   visualization for analysts, the conversational interface for everyone
   else. A loan officer can describe a borrower in plain English in Claude
   Desktop and get the same risk tier."
3. **Sample prompt** (point at the card): "$12,000, 36-month, 11.5% interest,
   sub-grade B2, $58k income, FICO 720, renting, debt consolidation. Returns
   default probability, risk tier, and approve/decline."

**Transition:** "Now the live demo."

---

## Tab 7 — Predict a loan (2:30, Aayush)

**Goal:** the wow factor. This is what the audience remembers. **Rehearse 5x.**

**Demo script — strict order:**

1. **Single mode, default profile.** Click `Predict default risk`. Point at
   gauge (~37%, High risk, Decline). Point at the why panel: "sub-grade
   pulling toward repayment, term 36 pulling toward repayment, RENT pushing
   toward default. Each bar is an XGBoost SHAP contribution."
2. **What-if mode.** Without re-clicking, drop FICO from 700 → 600. The
   gauge re-renders live. "Adjust any input and the prediction is real-time.
   No rebuild step."
3. **Compare mode.** Toggle "Compare two loans." Set Loan A preset = Safe,
   Loan B preset = Risky. Click `Predict both loans`. "Two gauges side by
   side — Safe at ~10%, Risky at ~60%. Both with their own why-panels."
4. **MCP live on Claude Desktop (the closer).** Switch to Claude Desktop
   (have it pre-opened in another window). Two prompts back to back:

   **Prompt A** (paste verbatim — same as Tab 6):
   > "Predict default risk for a $12,000, 36-month loan at 11.5% interest.
   > Sub-grade B2, $58,000 income, DTI 15%, FICO 720, renting, debt
   > consolidation."

   Tool fires, Claude returns 32% Moderate risk, Review.

   **Prompt B** (the wow moment):
   > "How can I get it approved?"

   Claude **autonomously runs 4-5 variations through the tool** ($8K loan,
   higher FICO, A5 grade, etc.) and synthesizes a comparison table from the
   results. This is multi-turn LLM reasoning over your model. Most teams
   won't have this - **lean into it.**

   Stage tip: keep the chat scrolled to Claude's natural-language response,
   not the raw tool-call JSON. The JSON proves it works; the prose is what
   the audience remembers.

**Transition:** "Junwei brings it home — what we found and what's next."

**Backup plan if anything fails:**
- Streamlit crashed → fall back to `presentation/presentation_slides.pdf`
- Claude Desktop fails → skip step 4, say "the MCP setup is documented in
  the README" and move on. Don't fight a broken tool live on stage.
- Laptop dies → second teammate has the same checkout open on theirs

**Backup MCP prompts if you have time / questions go technical:**
- "Compare two loans: (1) $5,000 at 7%, A1 grade, FICO 780, $90K income
  vs (2) $25,000 at 22%, E4 grade, FICO 620, $35K income. Which is
  riskier?" — Claude calls the tool twice and reasons about the contrast.
- "What FICO score would this same borrower need to drop below 20%
  default probability?" — iterative tuning toward a target.

These are listed in Tab 6 of the app as well, in case you blank.

---

## Tab 8 — Insights + Next (1:30, Junwei)

**Goal:** close the technical story. Three beats, ~30s each.

1. **Sub-grade chart** (point at the bar chart): "Default rate ramps
   cleanly from 6% at A1 to 45% at G5. The model mostly refines what
   Lending Club's own grade already knows."
2. **Limitations** (point at the red-bordered cards): "Honest about what we
   did NOT solve — class imbalance left as-is, no validation past 2017, no
   fairness audit, FICO is our only credit signal."
3. **What's next** (point at the green-bordered cards): "Real-time scoring
   API, fairness + adverse-action layer, quarterly retraining cadence, more
   macro signals beyond unemployment."

**Transition:** "Thank you — we'll take questions now."

---

## Tab 9 — Q&A (1:00, All 3)

Open the Q&A tab so the closing graphic is on screen. The "Likely questions,
brief answers" section is the cheat sheet — pre-discussed, not improvised.

**Anticipated questions and pre-cooked answers** — speaker matches topic
ownership so whoever knows that part of the work answers.

- **"Why XGBoost over HistGradientBoosting?"** (Lusi) — "AUC tied at the
  4th decimal, XGBoost was 5× faster to fit, exposes native SHAP."
- **"Why didn't extra macro features improve the model?"** (Aayush) — "The
  production pipeline already has UNRATE via MacroJoiner. The extra FRED
  features are correlated with each other, so they add little marginal
  signal in a linear model."
- **"Can this be deployed?"** (Aayush) — "MCP server is the deployment
  blueprint. FastAPI is the natural next step."
- **"Why no class imbalance correction?"** (Junwei) — "Tree models handle
  imbalance natively via `scale_pos_weight`. Resampling discards data and
  bakes a specific cost ratio into the model."
- **"How does the threshold get chosen?"** (Lusi) — "Youden's J on val,
  but the threshold tuner shows the operating point can be moved live to
  match a lender's cost ratio."

If you don't know an answer, **say so honestly** — don't invent. "Good
question, we didn't measure that — happy to follow up."

---

## Pre-presentation checklist (the morning of)

- [ ] All 3 team members have the repo cloned + `streamlit run` works on
      their laptop (in case the primary laptop fails)
- [ ] `presentation/presentation_slides.pdf` exists and opens (backup)
- [ ] `data/processed/preprocessing_pipeline.pkl` and
      `models/best_model.pkl` are present
- [ ] Theme is locked in (the `.streamlit/active_theme.txt` is committed)
- [ ] Streamlit app launches cleanly: `streamlit run src/app/streamlit_app.py`
- [ ] Predictor tab tested with all 3 presets (Safe / Borderline / Risky)
- [ ] Threshold tuner drag-tested at thresholds 0.30 / 0.50 / 0.70
- [ ] Compare mode tested with Safe vs Risky presets
- [ ] At least one full 15-minute dry run completed with stopwatch
- [ ] Each speaker has timed their solo segments and is within 10s of
      target
- [ ] If demoing MCP: Claude Desktop is signed in and the connector loads

## What "professional delivery" looks like

- Speak to the room, not the laptop screen.
- Don't read the bullets verbatim — paraphrase. The screen is the slide,
  your voice is the talk.
- Hand the clicker to whoever is speaking. No competing keyboards.
- If you go off-script, the next speaker still picks up at their tab.
- Land within the 15-minute window. The 1/2-hour slot includes Q&A and
  buffer; a tight 14:30 leaves room for unscripted questions.
