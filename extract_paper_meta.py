"""FrontierAtlas paper metadata extractor.

Day 2 additions:
  Task 1 - robust PDF fetching + cleaner header fields (title/authors/abstract/date)
  Task 2 - official-repo scoring, models/datasets from body text, tasks/methods
  Task 3 - SOTA claims from the abstract and from pdfplumber result tables
"""
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import requests

try:
    import pdfplumber
except ImportError:  # table extraction is optional
    pdfplumber = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("extract")

PDF_DIR = Path("pdfs")
OUT_FILE = Path("extracted_papers.json")
ARXIV_IDS = ["2503.18382", "2510.13998", "2607.18859", "2507.23370", "2503.11651"]
PDF_MIRRORS = [
    "https://arxiv.org/pdf/{id}.pdf",
    "https://export.arxiv.org/pdf/{id}",
    "https://arxiv.org/pdf/{id}v1",
]

# ---------------------------------------------------------------- constants
THIRD_PARTY = {
    "pytorch/pytorch", "huggingface/transformers", "tensorflow/tensorflow",
    "openai/openai-python", "numpy/numpy", "scipy/scipy", "facebookresearch/detectron2",
    "huggingface/datasets", "huggingface/peft", "vllm-project/vllm",
    "princeton-nlp/swe-bench", "swe-bench/swe-bench", "openai/human-eval",
    "huggingface/accelerate", "microsoft/deepspeed", "nvidia/apex",
    "facebookresearch/xformers", "opencv/opencv", "pandas-dev/pandas",
}
CODE_HINTS = re.compile(
    r"(code|implementation|source code|released?|available at|open-?source|our (?:repo|project|model))", re.I)

KNOWN_MODELS = [
    r"GPT-?4o(?:-mini)?", r"GPT-?4(?:\.\d)?", r"GPT-?5", r"o[134]-mini",
    r"Claude[\s-]?(?:3(?:\.5|\.7)?|4)[\s-]?(?:Sonnet|Opus|Haiku)",
    r"Gemini[\s-]?\d(?:\.\d)?[\s-]?(?:Pro|Flash)",
    r"Qwen-?\d?(?:\.\d)?(?:-Coder)?-\d+B(?:-Instruct)?",
    r"Llama-?\s?\d(?:\.\d)?-\d+B", r"DeepSeek-(?:V\d|R\d)(?:\.\d)?",
    r"BitNet(?:-b1\.58)?(?:-\d+(?:\.\d+)?B)?", r"UniMERNet", r"DUSt3R", r"MASt3R",
    r"VGGT", r"DINOv2", r"Trae Agent", r"PP-?FormulaNet(?:-[LSM])?",
    r"SWE-agent", r"Agentless", r"OpenHands", r"AutoCodeRover", r"CodeAct",
    r"Pi3", r"CUT3R", r"Fast3R", r"MonST3R", r"CoTracker", r"DepthAnything",
]
KNOWN_DATASETS = [
    "SWE-bench Verified", "SWE-bench Lite", "SWE-bench Multimodal", "SWE-bench",
    "HumanEval", "MATH-500", "GSM8K", "MMLU", "GLUE", "FineWeb", "C4",
    "ScanNet", "CO3D", "Co3Dv2", "RealEstate10K", "ETH3D", "DTU", "Re10K", "Tanks and Temples",
    "Human3.6M", "ScanNet++", "ARKitScenes", "Hypersim", "KITTI", "NYU-v2", "Sintel",
    "UniMER-1M", "UniMER-Test", "UniMERNet Benchmark", "CROHME", "HME100K", "SCE", "SPE",
    "Formula Mining System Dataset",
]
TASK_KEYWORDS = {
    "document-understanding": ["document", "ocr", "formula recognition", "layout"],
    "computer-vision": ["image", "visual", "vision", "3d", "camera"],
    "3d-reconstruction": ["3d reconstruction", "camera pose", "point map", "depth estimation"],
    "software-engineering": ["swe-bench", "software engineering", "issue resolution", "program repair",
                             "code repair", "repository-level"],
    "natural-language-processing": ["language model", "llm", "text generation"],
    "model-compression": ["quantization", "1.58-bit", "ternary", "distillation", "low-bit"],
}
METHOD_KEYWORDS = {
    "deep-learning": ["neural", "transformer", "deep learning", "llm"],
    "ocr": ["ocr", "text recognition", "formula recognition"],
    "distillation": ["distillation", "distill"],
    "quantization": ["quantization", "1.58-bit", "ternary"],
    "agents": ["agent"],
    "test-time-scaling": ["test-time scaling", "test-time compute"],
    "reinforcement-learning": ["reinforcement learning"],
    "robustness": ["robust"],
    "feed-forward-inference": ["feed-forward"],
}


# ---------------------------------------------------------- cleaning helpers
def clean_latex(s: str) -> str:
    s = re.sub(r"\\(?:textbf|textit|emph|mathrm|mathbf|text|textsc)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$[^$]*\$", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def clean_title(s: str) -> str:
    return clean_latex(s).strip(" .")


SUPERS = r"[\u00b9\u00b2\u00b3\u2070-\u209f\u2020\u2021\u00a7\u00b6*#\u2217]"


def clean_authors(raw: str) -> list:
    raw = re.sub(r"\S+@\S+", "", raw)
    raw = re.sub(r"\{[^}]*\}", "", raw)          # {a,b}@uni.edu style leftovers
    raw = re.sub(SUPERS, "", raw)
    raw = re.sub(r"\^?\{?\d+(?:,\d+)*\}?", "", raw)
    raw = re.sub(r"\band\b", ",", raw)
    names = [re.sub(r"\s+", " ", n).strip(" ,.") for n in re.split(r",|\n|;", raw)]
    seen, out = set(), []
    for n in names:
        if 2 < len(n) < 40 and 2 <= len(n.split()) <= 4 and re.fullmatch(r"[^\d@]+", n) \
                and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def dehyphenate(s: str) -> str:
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    return re.sub(r"\s+", " ", s).strip()


def body_text(text: str) -> str:
    """Text without the reference list (so cited repos/datasets are not picked up)."""
    cuts = [m.start() for m in re.finditer(r"\n\s*(?:References|REFERENCES|Bibliography)\s*\n", text)]
    cuts = [c for c in cuts if c > len(text) * 0.3]
    return text[:cuts[-1]] if cuts else text


# ------------------------------------------------------------- PDF access
def get_pdf(arxiv_id: str):
    """Use local pdfs/<id>.pdf if present; otherwise try several mirrors."""
    PDF_DIR.mkdir(exist_ok=True)
    path = PDF_DIR / f"{arxiv_id}.pdf"
    if path.exists() and path.stat().st_size > 1_000:
        return path
    for tpl in PDF_MIRRORS:
        url = tpl.format(id=arxiv_id)
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            if r.ok and r.content[:4] == b"%PDF":
                path.write_bytes(r.content)
                return path
        except Exception as e:
            log.debug("mirror failed %s: %s", url, e)
    log.warning("Could not download %s. Save the PDF manually as %s", arxiv_id, path)
    return None


def read_pdf(path: Path):
    doc = fitz.open(path)
    pages = [p.get_text() for p in doc]
    text = "\n".join(pages)
    # re-join GitHub URLs broken across lines
    text = re.sub(r"(github\.com/[\w.-]+/)\s*\n\s*", r"\1", text)
    text = re.sub(r"(github\.com/)\s*\n\s*", r"\1", text)
    return doc, text


# --------------------------------------------------- Task 1: header fields
def extract_title(doc) -> str:
    """Title = largest-font text on page 1."""
    try:
        spans = [s for b in doc[0].get_text("dict")["blocks"] if "lines" in b
                 for l in b["lines"] for s in l["spans"] if s["text"].strip()]
        top = max(s["size"] for s in spans)
        title = " ".join(s["text"].strip() for s in spans if abs(s["size"] - top) < 0.5)
        return clean_title(title)
    except Exception as e:
        log.warning("title failed: %s", e)
        return ""


def extract_authors(doc, title: str) -> list:
    try:
        text = doc[0].get_text()
        tail = title[-25:]
        after = text.split(tail, 1)[-1] if tail and tail in text else text
        block = re.split(r"\bAbstract\b", after, maxsplit=1, flags=re.I)[0]
        block = "\n".join(l for l in block.splitlines() if not re.search(
            r"arxiv|universit|institut|research|lab\b|inc\.|corp|department|school|college|@",
            l, re.I))
        return clean_authors(block)
    except Exception as e:
        log.warning("authors failed: %s", e)
        return []


def extract_abstract(text: str) -> str:
    patterns = [
        r"\bAbstract\b[\s\-\u2014:.]*(.+?)(?:\n\s*(?:1\.?\s*)?Introduction|\bKeywords\b|\bIndex Terms\b)",
        r"\bAbstract\b[\s\-\u2014:.]*(.{200,2500}?)\n\s*\n",
    ]
    for p in patterns:
        m = re.search(p, text, re.S | re.I)
        if m and len(m.group(1)) > 100:
            return dehyphenate(m.group(1))
    return ""


def extract_date(text: str) -> str:
    m = re.search(r"arXiv:\d{4}\.\d{4,5}v\d+\s+\[[^\]]+\]\s+(\d{1,2}\s+\w{3}\s+\d{4})", text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d %b %Y").strftime("%Y%m%d")
        except ValueError:
            pass
    log.warning("no arXiv date stamp found")
    return ""


# ------------------------------------------- Task 2: repo, models, datasets
def normalize_github(url: str):
    m = re.match(r"https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)", url)
    if not m:
        return None
    org, repo = m.group(1), re.sub(r"\.git$", "", m.group(2)).rstrip(".,;)")
    if org.lower() in {"features", "orgs", "topics", "sponsors", "settings"}:
        return None
    return f"https://github.com/{org}/{repo}"


def extract_github(text: str, abstract: str, first_page: str):
    """Return (url, score). Dependencies are dropped; links in the abstract / first page /
    next to 'code is available' wording score highest; reference-list links are penalised."""
    ref_start = len(body_text(text))
    third = {t.lower() for t in THIRD_PARTY}
    best, best_score = None, 0
    for m in re.finditer(r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+[^\s)\]]*", text):
        url = normalize_github(m.group(0))
        if not url or "/".join(url.split("/")[-2:]).lower() in third:
            continue
        ctx = text[max(0, m.start() - 150): m.start()]
        score = 1
        score += 3 if CODE_HINTS.search(ctx) else 0
        score += 4 if (m.group(0) in abstract or m.group(0) in first_page) else 0
        score -= 4 if m.start() >= ref_start else 0
        if score > best_score:
            best, best_score = url, score
    return (best, best_score) if best_score >= 3 else (None, 0)


def title_name(title: str) -> str:
    m = re.match(r"^([A-Za-z0-9][\w\-\.]*)\s*:", title)
    return m.group(1) if m else ""


def extract_models(text: str, title: str) -> list:
    found = {}
    names = [title_name(title)] if title_name(title) else []
    for pat in KNOWN_MODELS:
        names += [m.group(0) for m in re.finditer(pat, text)]
    for n in names:
        n = n.strip(" -")
        if n.lower().startswith("pp-formulanet"):
            n = "PPFormulaNet" + n[len("PP-FormulaNet"):]
        found.setdefault(re.sub(r"[\s\-]", "", n).lower(), n)
    vals = list(found.values())
    vals = [v for v in vals if not any(o != v and o.lower().startswith(v.lower())
                                        and o[len(v)].isalpha() for o in vals)]
    return sorted(vals)


def extract_datasets(text: str) -> list:
    out = []
    for d in KNOWN_DATASETS:
        if re.search(r"(?<![\w-])" + re.escape(d) + r"(?![\w-])", text, re.I):
            # skip a shorter name when a longer one that contains it is already found
            if not any(d.lower() in o.lower() for o in out):
                out.append(d)
    return out


def keyword_labels(table: dict, blob: str) -> list:
    blob = blob.lower()
    return [label for label, kws in table.items() if any(k in blob for k in kws)]


# ------------------------------------------------- Task 3: SOTA claims
TRIGGER = re.compile(r"outperform|faster|improv|reduc|surpass|speedup|state-of-the-art|sota|"
                     r"achiev|resolv|less memory|higher", re.I)
VALUE = re.compile(r"(\d+(?:\.\d+)?\s?(?:%|x|\u00d7))(\s+faster)?", re.I)
TARGET = re.compile(r"(?:than|over|against|compared (?:to|with)|outperforms?|outperforming|"
                    r"surpass(?:es|ing)?|beats?|versus|vs\.?)\s+(?:the\s+)?"
                    r"([A-Z][\w\-\.]*(?:\s[A-Z0-9][\w\-\.]*)?)")


def sentence_metric(sent: str):
    s = sent.lower()
    if "memory" in s:
        return "Memory Reduction", "Memory Footprint"
    if "latency" in s:
        return "Latency Reduction", "Inference Speed"
    if any(k in s for k in ("faster", "speedup", "speed", "throughput", "inference")):
        return "Speedup Factor", "Inference Speed"
    if "resolve" in s or "resolution rate" in s:
        return "Resolve Rate", ""
    if "pass@1" in s:
        return "Pass@1", ""
    if "f1" in s:
        return "F1 Score", ""
    if "accuracy" in s or "accurate" in s:
        return "Accuracy", ""
    return "Improvement", ""


def abstract_claims(abstract: str, models: list, self_name: str = "") -> list:
    claims = []
    others = [m for m in models if not (self_name and m.lower().startswith(self_name.lower()))]
    for sent in re.split(r"(?<=[.!?])\s+", abstract):
        if not TRIGGER.search(sent):
            continue
        prev_end = 0
        for m in VALUE.finditer(sent):
            # local window: text since the previous value (or sentence start) up to just after this one
            window = sent[prev_end: m.end() + 40]
            after = sent[m.end():]
            prev_end = m.end()
            metric, default_bench = sentence_metric(window)
            bench = next((d for d in KNOWN_DATASETS if d.lower() in sent.lower()), default_bench)
            target = next((o for o in others if o.lower() in after.lower()), "")
            if not target:
                t = TARGET.search(after)
                target = t.group(1).strip() if t else ""
            if not target:
                target = next((o for o in others if o.lower() in sent.lower()), "")
            value = m.group(1).replace(" ", "").replace("\u00d7", "x")
            if m.group(2):
                value += " faster"
            claims.append({"benchmark": bench, "metric": metric, "value": value,
                           "comparison_target": target})
    return claims


METRIC_HEADER = re.compile(r"pass@\d|resolve|resolved|acc|f1|bleu|exp|cdm|edit|miou|auc|psnr|ssim|"
                           r"score|error|err|chamfer|rate|\bap\b|map", re.I)
LOWER_BETTER = re.compile(r"err|edit|chamfer|latency|loss|wer|cer|memory", re.I)
NUM = re.compile(r"^\s*\**(\d+(?:\.\d+)?)\s*%?\**\s*$")


def table_claims(path: Path, self_name: str, limit: int = 6) -> list:
    """For each table metric column: proposed method (row containing the paper's own name,
    else best row) vs the strongest other row."""
    claims = []
    if pdfplumber is None:
        return claims
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text() or ""
                cap = re.search(r"Table\s+\d+[.:]\s*([^\n]{0,80})", ptext)
                for tb in page.extract_tables():
                    if len(tb) < 4 or len(tb[0]) < 2:
                        continue
                    header = [(c or "").replace("\n", " ").strip() for c in tb[0]]
                    for ci, h in enumerate(header):
                        if ci == 0 or not METRIC_HEADER.search(h):
                            continue
                        rows = []
                        for r in tb[1:]:
                            if ci < len(r) and r[0] and r[ci]:
                                m = NUM.match(r[ci].replace("\n", " "))
                                if m:
                                    rows.append((r[0].replace("\n", " ").strip(), float(m.group(1)), r[ci]))
                        if len(rows) < 3:
                            continue
                        low = bool(LOWER_BETTER.search(h))
                        rows_sorted = sorted(rows, key=lambda x: x[1], reverse=not low)
                        ours = next((r for r in rows if self_name and self_name.lower() in r[0].lower()),
                                    rows_sorted[0])
                        others = [r for r in rows_sorted if r is not ours]
                        if not others:
                            continue
                        pct = "%" if ("%" in h or "%" in ours[2]) else ""
                        bench = next((d for d in KNOWN_DATASETS if d.lower() in ptext.lower()), "")
                        claims.append({
                            "benchmark": bench or (cap.group(1).strip() if cap else ""),
                            "metric": h,
                            "value": f"{ours[1]:g}{pct}",
                            "comparison_target": others[0][0],
                        })
                        if len(claims) >= limit:
                            return claims
    except Exception as e:
        log.warning("table extraction failed: %s", e)
    return claims


def merge_claims(*groups, cap: int = 10) -> list:
    seen, out = set(), []
    for g in groups:
        for c in g:
            key = (c["value"], c["benchmark"].lower(), c["metric"].lower())
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out[:cap]


# ------------------------------------------------------------- pipeline
def process(arxiv_id: str) -> dict:
    rec = {
        "arxiv_id": arxiv_id, "title": "", "authors": [], "publication_date": "",
        "arxiv_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "arxiv_page_url": f"https://arxiv.org/abs/{arxiv_id}",
        "github_url": None, "is_official_code": False, "tasks": [], "methods": [],
        "models": [], "datasets": [], "sota_claims": [], "abstract": "",
    }
    path = get_pdf(arxiv_id)
    if not path:
        return rec
    try:
        doc, text = read_pdf(path)
    except Exception as e:
        log.error("Cannot open %s: %s", path, e)
        return rec

    body = body_text(text)
    steps = [
        ("title", lambda: rec.update(title=extract_title(doc))),
        ("authors", lambda: rec.update(authors=extract_authors(doc, rec["title"]))),
        ("abstract", lambda: rec.update(abstract=extract_abstract(text))),
        ("date", lambda: rec.update(publication_date=extract_date(text))),
        ("github", lambda: _set_github(rec, text, doc)),
        ("models", lambda: rec.update(models=extract_models(body, rec["title"]))),
        ("datasets", lambda: rec.update(datasets=extract_datasets(body))),
        ("tasks/methods", lambda: rec.update(
            tasks=keyword_labels(TASK_KEYWORDS, rec["title"] + " " + rec["abstract"]),
            methods=keyword_labels(METHOD_KEYWORDS, rec["title"] + " " + rec["abstract"]))),
        ("sota", lambda: rec.update(sota_claims=merge_claims(
            abstract_claims(rec["abstract"], rec["models"], title_name(rec["title"])),
            table_claims(path, title_name(rec["title"]))))),
    ]
    for name, fn in steps:          # one failing field never stops the others
        try:
            fn()
        except Exception as e:
            log.error("[%s] %s failed: %s", arxiv_id, name, e)
    return rec


def _set_github(rec, text, doc):
    url, score = extract_github(text, rec["abstract"], doc[0].get_text())
    rec["github_url"] = url
    rec["is_official_code"] = bool(url) and score >= 4


def main(ids):
    results = [process(i) for i in ids]
    OUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", OUT_FILE)


if __name__ == "__main__":
    main(sys.argv[1:] or ARXIV_IDS)
