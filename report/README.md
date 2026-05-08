# Phase 4 Report

LaTeX source for the Phase 4 IEEE-format report. Files in this folder:

- `Phase4Report.tex` - main source, uses `IEEEtran.cls`
- `references.bib` - BibTeX entries
- `figures/` - PNGs the report uses

## Build it

Easiest is Overleaf if you don't have a TeX install:

1. Make a free account at https://www.overleaf.com
2. New Project -> Upload Project, zip up this folder and upload
3. Set the main document to `Phase4Report.tex`, compiler to pdfLaTeX
4. Recompile. Overleaf does pdflatex -> bibtex -> pdflatex -> pdflatex on its own so the citations resolve.

If you have TeX Live or MiKTeX locally:

```bash
cd report
pdflatex Phase4Report.tex
bibtex   Phase4Report
pdflatex Phase4Report.tex
pdflatex Phase4Report.tex
```

Output is `Phase4Report.pdf`. Thats the file we submit.
