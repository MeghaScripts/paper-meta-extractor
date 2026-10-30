# FrontierAtlas – Paper OCR & Deep Metadata Extraction

`extract_paper_meta.py` parses arXiv preprint PDFs with **PyMuPDF** and emits `extracted_papers.json`
(title, authors, abstract, publication_date, official `github_url`, models, datasets, `sota_claims`, tasks, methods).

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
# the 5 test papers: downloads arxiv.org/pdf/<id>.pdf into ./papers/ (or put the PDFs there yourself)
python extract_paper_meta.py --out extracted_papers.json

python extract_paper_meta.py --ids 2503.18382 2507.23370     # specific papers
python extract_paper_meta.py --pdf-dir ./my_pdfs             # every <id>.pdf in a folder
export ANTHROPIC_API_KEY=...; python extract_paper_meta.py --llm   # LLM-assisted models/datasets/claims
python extract_paper_meta.py -v                              # show GitHub candidate scores
python extract_paper_meta.py --no-api                        # PDF only, skip the arXiv API cross-check
python tests/test_extract.py                                 # 17 offline tests
```
PDFs must be named `<arxiv_id>.pdf`. Without `--llm` everything is deterministic (regex + layout heuristics, no network
besides the optional PDF download).

## How it works
| Field | Method |
|---|---|
| title | Largest-font contiguous lines at the top of page 1; LaTeX stripped |
| authors | Lines after the title until affiliations/"Abstract"; split on `,`/`and`; digits, `*†‡`, unicode superscripts and e-mails removed |
| abstract | Text between "Abstract" and "Introduction"; de-hyphenated and un-wrapped; ligatures fixed |
| publication_date | arXiv side stamp (`arXiv:… [cs.CV] 24 Mar 2025`) → `YYYYMMDD`; falls back to PDF creation date |
| models / datasets | Default: curated regex families + paper-name variants (e.g. `PP-FormulaNet-L/S`) + "X dataset/benchmark" pattern, bibliography excluded. `--llm`: structured JSON from Claude |
| sota_claims | Numeric claims (`%`, `x`, `×`) from the abstract; metric = keyword *closest to the number*, baseline = target of "than/over/against/outperforms…" |

**arXiv API cross-check.** When online, title / authors / abstract / date parsed from the PDF are compared with the official arXiv API
record; empty or clearly disagreeing fields (title similarity < 0.85, author surnames overlap < 0.6, abstract similarity < 0.9) are
replaced by the API values, so a layout the PDF parser mishandles still yields a clean record. Offline, the PDF parse stands alone.

Every stage is wrapped in `safe()`; a missing/corrupt PDF or an unparseable field yields `null`/`[]`, never a crash.

## How the official GitHub repo is chosen
Every `github.com/<owner>/<repo>` in the PDF (text *and* clickable link annotations, so URLs broken across lines are
recovered) is normalised to `https://github.com/owner/repo` (no `.git`, `/tree/…`, `/blob/…`, `/issues`) and **scored by
where and how it is mentioned**. Bonuses: it appears in the abstract or first-page footnote (+6/+3), the context says
"code / models / released / publicly available" (+3), the repo name overlaps the paper title (+3), the owner matches an
author or affiliation (+2), it is mentioned repeatedly (+). Penalties: it appears only in the References (−6), the context
says "based on / following / adopted from / baseline / pip install" (−3), or it is a known dependency such as
`pytorch/pytorch` or `huggingface/transformers` (−20). The top-scoring candidate wins only if it clears a minimum score,
so a paper with no released code returns `github_url: null` and `is_official_code: false`. Ties (e.g. PaddleOCR vs PaddleX,
both in the abstract) go to the first mention.

## Verification status
* The script was run on the 5 real PDFs (output reviewed line by line). That run showed five issues - author order in two-column
  author blocks, a group name ("Trae Research Team") in the author list, claims with missing baselines / wrong metrics, noisy model
  and dataset names, and footnote digits in an abstract - all fixed and covered by regression tests built from the real output
  (`tests/fixtures/real_run_1.json`). 17 offline tests: `python tests/test_extract.py`.
* **Re-run once after unpacking** to regenerate `extracted_papers.json` with the final script:
  ```bash
  pip install -r requirements.txt && python extract_paper_meta.py -v
  ```
* Limits: `sota_claims` from the default (non-LLM) mode come from the abstract; table-level results are best captured with
  `--llm`. The arXiv title is "PP-FormulaNet" (hyphenated), not "PPFormulaNet".

## Files
`extract_paper_meta.py` · `requirements.txt` · `extracted_papers.json` · `README.md` · `tests/test_extract.py` · `tests/fixtures/real_run_1.json`
