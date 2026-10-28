# FrontierAtlas Paper Metadata Extractor

Extracts title, authors, abstract, date, official GitHub repo, models, datasets
and SOTA claims from arXiv preprints (PyMuPDF + regex).

## Setup
    python -m venv venv
    venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
    pip install -r requirements.txt

## Run
    python extract_paper_meta.py                 # all 5 test papers
    python extract_paper_meta.py 2503.11651      # single paper

PDFs are read from `pdfs/<arxiv_id>.pdf`. If the automatic download fails,
download them manually from arxiv.org/pdf/<id> and save them in `pdfs/`.

## Output
`extracted_papers.json` (FrontierAtlas schema).

## Official vs third-party GitHub links
All GitHub URLs in the PDF are normalized and a blocklist of common
dependencies (pytorch, transformers, etc.) is dropped. The remaining links are
scored: higher if they appear in the abstract/first page or right after wording
such as "code is available" or "our implementation". The top-scoring link is
returned, or `null` if none remain.

## Status
Work in progress: title/author/abstract/GitHub logic started; model, dataset
and SOTA extraction are first-pass heuristics.
