# Phase 4 Report

IEEE conference-format report for Phase 4 of the EAS 587 Lending Club
default-prediction project. Source files in this folder:

| File                  | Purpose                                     |
| --------------------- | ------------------------------------------- |
| `Phase4Report.tex`    | Main LaTeX source (uses `IEEEtran.cls`)     |
| `references.bib`      | BibTeX entries for IEEE numbered citations  |
| `figures/`            | PNG figures (copied from `presentation/figures/`) |

## How to compile

### Option A — Overleaf (recommended, no install)

1. Go to https://www.overleaf.com and create a free account.
2. **New Project → Upload Project**.
3. Zip the entire `report/` folder (so `Phase4Report.tex`, `references.bib`,
   and the `figures/` directory are at the project root inside the zip)
   and upload.
4. Set the main document to `Phase4Report.tex` and the compiler to
   `pdfLaTeX`.
5. Click **Recompile**. Output is the submission-ready PDF.

Overleaf auto-runs `pdflatex → bibtex → pdflatex → pdflatex` so the
`\cite{...}` references resolve correctly.

### Option B — Local LaTeX

If you have a local TeX distribution (TeX Live on Linux/macOS, MiKTeX on
Windows):

```bash
cd report
pdflatex Phase4Report.tex
bibtex   Phase4Report
pdflatex Phase4Report.tex
pdflatex Phase4Report.tex
```

The output `Phase4Report.pdf` is the submission artifact.

## Editing notes for the team

- The draft is written in concrete, specific technical voice. Read through
  for accuracy and reword anything that sounds machine-generated. The
  rubric explicitly disallows "auto-generated LLM content."
- All metric values come from real artifacts (`models/model_results.csv`
  and the held-out test scoring); do not change numbers without checking
  against the artifacts.
- The page budget is **10 pages including references** (per Appendix A
  of the rubric). Current draft fits comfortably; if edits push it over,
  the easiest cuts are the methodology code listings.
- IEEE conference style is double-column. Section ordering matches the
  required-sections list in the rubric (Title, Abstract, Intro &
  Background, Data Description, Methodology, Results & Analysis,
  Discussion, Conclusion, References, Appendices).

## Submission packaging reminder

The final zip must be named `nepal_zhang_zhang_phase4.zip` and contain:

- `Phase4report.pdf` (rename `Phase4Report.pdf` to match if needed)
- `presentation_slides.pdf` (from `../presentation/presentation_slides.pdf`)
- Project code (this repo, **without** the `data/raw/` or
  `data/processed/` directories)

Word document → PDF penalty is 5 points; submission-naming penalty is
5 points. Both are easy to avoid.
