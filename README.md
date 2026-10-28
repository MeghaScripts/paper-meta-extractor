# FrontierAtlas Paper Metadata Extractor

Extracts title, authors, abstract, publication date, official GitHub repo, models, datasets,
tasks/methods and SOTA claims from arXiv preprints using PyMuPDF, pdfplumber and regex.

## Setup
    python -m venv venv
    venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
    pip install -r requirements.txt

## Run
    python extract_paper_meta.py                 # all 5 test papers
    python extract_paper_meta.py 2503.11651      # single paper
    python verify_output.py                      # quality report on extracted_papers.json

PDFs are read from `pdfs/<arxiv_id>.pdf`. If the automatic download (arxiv.org and mirrors)
fails, download them in a browser from `https://arxiv.org/pdf/<id>` and save them in `pdfs/`.

## Output
`extracted_papers.json` (FrontierAtlas schema). A failure in one field never stops the others;
missing fields are left empty / `null`.

## How the pipeline works
- **Header**: title = largest font on page 1; authors cleaned of digits, symbols and emails;
  abstract de-hyphenated; date from the arXiv stamp (`24 Mar 2025` -> `20250324`).
- **Models / datasets**: matched in the body text only (the reference list is cut off), so cited
  works are not counted. Title name (e.g. `PPFormulaNet`) is added as a model.
- **SOTA claims**: numeric claims from abstract sentences (metric and baseline inferred from the
  wording) plus table rows read with pdfplumber (the paper's own row vs the strongest other row).

## Official vs third-party GitHub links
Every GitHub URL in the PDF is normalised to `https://github.com/org/repo` and a blocklist of
common dependencies (pytorch, transformers, ...) is dropped. Remaining links are scored: higher if
they appear in the abstract or on the first page, or right after wording such as "code is
available" or "our implementation"; links that only appear in the reference list are penalised.
The best link is returned (and `is_official_code` is true when it has strong evidence), otherwise
`null`.

## Known limitations
Model/dataset lists use a curated pattern list and may need extending; table claims are heuristic
and should be spot-checked with `verify_output.py`.
