"""Quick quality check for extracted_papers.json (Task 1: verify the 5 papers).

Usage: python verify_output.py
Prints a per-paper report and flags anything that looks wrong.
"""
import json
import re
import sys
from pathlib import Path

REQUIRED = ["arxiv_id", "title", "authors", "publication_date", "arxiv_pdf_url", "arxiv_page_url",
            "github_url", "is_official_code", "tasks", "methods", "models", "datasets",
            "sota_claims", "abstract"]


def check(rec: dict) -> list:
    issues = []
    for k in REQUIRED:
        if k not in rec:
            issues.append(f"missing key: {k}")
    if not rec.get("title") or re.search(r"[\\$]", rec["title"]):
        issues.append("title empty or contains LaTeX")
    authors = rec.get("authors", [])
    if not authors:
        issues.append("no authors")
    if any(re.search(r"[\d@*\u2020\u2021\u00b9\u00b2\u00b3]", a) for a in authors):
        issues.append("author names contain digits/symbols/emails")
    if len(rec.get("abstract", "")) < 200:
        issues.append("abstract missing or too short")
    if re.search(r"\w-\s", rec.get("abstract", "")) and "\n" in rec.get("abstract", ""):
        issues.append("abstract has line breaks")
    if not re.fullmatch(r"\d{8}", rec.get("publication_date", "")):
        issues.append("publication_date is not YYYYMMDD")
    g = rec.get("github_url")
    if g is not None and not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", g):
        issues.append("github_url format invalid")
    if g and (g.endswith(".git") or "/tree/" in g or "/issues" in g):
        issues.append("github_url not clean")
    if not rec.get("models"):
        issues.append("no models")
    if not rec.get("datasets"):
        issues.append("no datasets")
    for c in rec.get("sota_claims", []):
        if set(c) != {"benchmark", "metric", "value", "comparison_target"}:
            issues.append("sota claim has wrong keys")
            break
        if not c["value"] or not c["metric"]:
            issues.append("sota claim with empty metric/value")
            break
    if not rec.get("sota_claims"):
        issues.append("no sota_claims")
    return issues


def main(path="extracted_papers.json"):
    p = Path(path)
    if not p.exists():
        sys.exit(f"{path} not found - run extract_paper_meta.py first")
    data = json.loads(p.read_text(encoding="utf-8"))
    bad = 0
    for rec in data:
        print(f"\n=== {rec.get('arxiv_id')} | {rec.get('title', '')[:70]}")
        print(f"authors: {rec.get('authors')}")
        print(f"date: {rec.get('publication_date')}  github: {rec.get('github_url')} "
              f"(official={rec.get('is_official_code')})")
        print(f"models: {rec.get('models')}")
        print(f"datasets: {rec.get('datasets')}")
        print(f"sota_claims: {len(rec.get('sota_claims', []))}")
        issues = check(rec)
        bad += bool(issues)
        for i in issues:
            print(f"  [!] {i}")
        if not issues:
            print("  OK")
    print(f"\n{len(data) - bad}/{len(data)} papers passed all checks")


if __name__ == "__main__":
    main(*sys.argv[1:])
