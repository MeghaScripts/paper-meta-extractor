#!/usr/bin/env python3
"""
extract_paper_meta.py - FrontierAtlas paper metadata extractor.

Parses arXiv preprint PDFs (PyMuPDF) and extracts:
  title, authors, abstract, publication_date, github_url (official code only),
  models, datasets, sota_claims  (+ tasks / methods tags).

Pipeline per paper:
  1. locate PDF  (./papers/<id>.pdf  ->  else download from arxiv.org)
  2. layout-aware header parse (font sizes on page 1) -> title / authors
  3. section parse -> abstract (de-hyphenated, un-wrapped)
  4. arXiv side-stamp / PDF metadata -> publication_date (YYYYMMDD)
  5. GitHub candidates scored by *where* and *how* they are mentioned
  6. models / datasets / SOTA claims:
       - LLM-assisted when ANTHROPIC_API_KEY is set (--llm), else
       - deterministic regex/heuristic extraction (default, no network)

Every field is wrapped so a failure yields null / [] instead of a crash.

Usage:
    python extract_paper_meta.py --ids 2503.18382 2510.13998 --out extracted_papers.json
    python extract_paper_meta.py --pdf-dir ./papers            # all PDFs in a folder
    python extract_paper_meta.py --llm                          # LLM-assisted entities
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:
    try:
        import fitz  # older PyMuPDF
    except ImportError:  # pragma: no cover
        sys.exit("PyMuPDF is required:  pip install pymupdf")

log = logging.getLogger("extract_paper_meta")

DEFAULT_IDS = ["2503.18382", "2510.13998", "2607.18859", "2507.23370", "2503.11651"]

# --------------------------------------------------------------------------- #
# Generic text cleaning
# --------------------------------------------------------------------------- #
SUPERSCRIPT_CHARS = "¹²³⁴⁵⁶⁷⁸⁹⁰†‡§¶∗*⋆★♠♣♦♥∥‖#"
LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}


def strip_latex(text: str) -> str:
    """Remove common LaTeX markup, keep readable text."""
    text = re.sub(r"\\(?:textbf|textit|emph|texttt|mathbf|mathrm|text|textsc)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\$\^\{?[^$}]*\}?\$", "", text)          # $^3$ , $^{1,2}$
    text = re.sub(r"\$([^$]*)\$", r"\1", text)               # inline math -> content
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)               # stray commands
    text = text.replace("{", "").replace("}", "")
    return text


def normalise_ws(text: str) -> str:
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def dehyphenate_and_join(text: str) -> str:
    """Join wrapped lines and repair 'recogni-\\ntion' style hyphenation."""
    text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)    # hard hyphen at line end
    text = re.sub(r"\s*\n\s*", " ", text)
    return normalise_ws(text)


# --------------------------------------------------------------------------- #
# PDF acquisition
# --------------------------------------------------------------------------- #
def get_pdf(arxiv_id: str, pdf_dir: Path) -> Path | None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    local = pdf_dir / f"{arxiv_id}.pdf"
    if local.exists() and local.stat().st_size > 500:
        return local
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "frontieratlas-extractor/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            local.write_bytes(r.read())
        return local
    except Exception as exc:  # network blocked, 404, ...
        log.warning("Could not download %s (%s). Put the PDF at %s", url, exc, local)
        return None


# --------------------------------------------------------------------------- #
# Header: title / authors
# --------------------------------------------------------------------------- #
def page_lines(page: "fitz.Page") -> list[dict]:
    """Return text lines on a page with size, y-position and text."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if abs(line.get("dir", (1, 0))[1]) > 0.3:      # rotated text, e.g. arXiv side stamp
                continue
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"])
            size = max(s["size"] for s in spans)
            out.append({"text": text, "size": size, "y": line["bbox"][1],
                        "x0": line["bbox"][0], "x1": line["bbox"][2]})
    out.sort(key=lambda l: (round(l["y"]), l["x0"]))
    return out


def extract_title(lines: list[dict]) -> tuple[str | None, int]:
    """Title = consecutive top-of-page lines in the largest font. Returns (title, last_idx)."""
    body = [l for l in lines if not re.match(r"^arXiv:", l["text"].strip())]
    if not body:
        return None, -1
    top = max(l["size"] for l in body[:15])
    idxs = [i for i, l in enumerate(lines) if abs(l["size"] - top) < 0.6 and l["y"] < 300
            and not l["text"].strip().lower().startswith("arxiv")]
    if not idxs:
        return None, -1
    # keep first contiguous run
    run = [idxs[0]]
    for i in idxs[1:]:
        if i - run[-1] <= 1:
            run.append(i)
        else:
            break
    title = " ".join(lines[i]["text"].strip() for i in run)
    title = normalise_ws(strip_latex(title))
    title = re.sub(r"\s+([:,])", r"\1", title)
    return title or None, run[-1]


AFFIL_WORDS = re.compile(
    r"(universit|institute|department|school|laborator|research|corporation|inc\.|ltd|"
    r"college|academy|google|microsoft|meta|baidu|bytedance|alibaba|tencent|@|\.edu|\.com|"
    r"\.org|\.cn|china|usa|united|oxford|beijing|shanghai)", re.I)


def clean_author(name: str) -> str:
    name = strip_latex(name)
    name = re.sub(r"[\w.+-]+@[\w.-]+", "", name)                       # emails
    name = re.sub(f"[{re.escape(SUPERSCRIPT_CHARS)}]", "", name)         # symbols
    name = re.sub(r"[\u2070-\u209f]", "", name)                          # unicode super/subscripts
    name = re.sub(r"(?<=[A-Za-z\u00c0-\u024f.])\d+(?:[,\d]*)", "", name) # trailing affil numbers
    name = re.sub(r"^\d+|\d+$", "", name.strip())
    name = re.sub(r"[,;]+$", "", name.strip())
    return normalise_ws(name)


GROUP_RE = re.compile(r"\b(team|group|lab|laborator(?:y|ies)|consortium|collaboration|research|inc|corp)\b", re.I)


def is_person_name(n: str) -> bool:
    return bool(n) and not GROUP_RE.search(n)


def extract_authors(lines: list[dict], start: int) -> list[str]:
    """Collect lines after the title until affiliation/abstract starts."""
    chunks: list[str] = []
    for l in lines[start + 1:start + 14]:
        t = l["text"].strip()
        if re.match(r"^abstract\b", t, re.I):
            break
        if AFFIL_WORDS.search(t) and not chunks:
            continue
        if AFFIL_WORDS.search(t):
            break
        chunks.append(t)
    raw = ", ".join(chunks)
    raw = re.sub(r"\s+and\s+", ", ", raw)
    # superscript/affiliation markers directly after a name act as separators ("Doe¹ Smith*²")
    raw = re.sub(rf"(?<=[A-Za-z\u00c0-\u024f.])[\d,{re.escape(SUPERSCRIPT_CHARS)}\u2070-\u209f]+(?=\s|,|$)", ",", raw)
    parts = [p for p in re.split(r"[,;]| {3,}", raw) if p.strip()]
    expanded = []
    for p in parts:      # "Hongen Liu Cheng Cui" (no separators) -> pairs of capitalised words
        w = clean_author(p).split()
        if len(w) >= 4 and len(w) % 2 == 0 and all(x[:1].isupper() for x in w):
            expanded += [" ".join(w[i:i + 2]) for i in range(0, len(w), 2)]
        else:
            expanded.append(p)
    authors, seen = [], set()
    for p in expanded:
        n = clean_author(p)
        if len(n) < 3 or not re.search(r"[A-Za-z]", n) or len(n.split()) > 5 or not is_person_name(n):
            continue
        if n.lower() not in seen:
            seen.add(n.lower())
            authors.append(n)
    return authors


# --------------------------------------------------------------------------- #
# Abstract
# --------------------------------------------------------------------------- #
def strip_footnote_marks(a: str) -> str:
    """'available at PaddleOCR1 and PaddleX2.' -> 'available at PaddleOCR and PaddleX.'"""
    m = re.search(r"\b(?:available|released|open-sourced)\s+at\s+", a, re.I)
    if not m:
        return a
    tail = re.sub(r"\b([A-Z][A-Za-z]{2,}[A-Za-z])([1-9])(?=\s+and\b|\s*[.,;]|\s*$)", r"\1", a[m.end():])
    return a[:m.end()] + tail


def extract_abstract(full_text: str, first_page_text: str) -> str | None:
    src = first_page_text if re.search(r"abstract", first_page_text, re.I) else full_text
    m = re.search(r"\babstract\b[\s\-—:.]*", src, re.I)
    if not m:
        return None
    rest = src[m.end():]
    end = re.search(r"\n\s*(?:1\.?\s+|I\.?\s+)?introduction\b|\n\s*keywords?\b|\n\s*index terms\b|"
                    r"\n\s*CCS concepts", rest, re.I)
    abstract = rest[:end.start()] if end else rest[:2500]
    abstract = re.sub(r"\n\s*\d+\s*\n", "\n", abstract)        # stray footnote markers
    abstract = dehyphenate_and_join(strip_latex(abstract))
    abstract = strip_footnote_marks(abstract)
    return abstract or None


# --------------------------------------------------------------------------- #
# Date
# --------------------------------------------------------------------------- #
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


STAMP_RE = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?\s*\[[^\]]+\]\s*(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})")


def _stamp_to_date(m) -> str | None:
    mon = MONTHS.get(m.group(2).lower())
    return f"{int(m.group(3)):04d}{mon:02d}{int(m.group(1)):02d}" if mon else None


def extract_date(doc: "fitz.Document", arxiv_id: str) -> str | None:
    """arXiv side stamp (possibly rotated / split over spans) -> PDF creation date."""
    if len(doc):
        page = doc[0]
        candidates = [re.sub(r"\s+", " ", page.get_text("text"))]
        for blk in page.get_text("dict")["blocks"]:           # includes rotated lines
            for ln in blk.get("lines", []):
                candidates.append("".join(s["text"] for s in ln["spans"]))
        candidates.append(" ".join(c for c in candidates[1:] if "arXiv" in c))
        for c in candidates:
            m = STAMP_RE.search(re.sub(r"\s+", " ", c))
            if m and _stamp_to_date(m):
                return _stamp_to_date(m)
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", (doc.metadata or {}).get("creationDate", "") or "")
    return "".join(m.groups()) if m else None


# --------------------------------------------------------------------------- #
# GitHub - official repository detection
# --------------------------------------------------------------------------- #
GH_RE = re.compile(r"github\.com\s*/\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*/\s*([A-Za-z0-9_.\-]+)", re.I)

# Libraries papers *use*; never the paper's own code.
DEPENDENCY_REPOS = {
    "pytorch/pytorch", "huggingface/transformers", "huggingface/datasets", "huggingface/peft",
    "huggingface/accelerate", "huggingface/trl", "tensorflow/tensorflow", "google/jax",
    "numpy/numpy", "scipy/scipy", "scikit-learn/scikit-learn", "pandas-dev/pandas",
    "openai/gym", "facebookresearch/detectron2", "open-mmlab/mmdetection", "vllm-project/vllm",
    "microsoft/deepspeed", "deepspeedai/deepspeed", "nvidia/apex", "ggerganov/llama.cpp",
    "princeton-nlp/swe-bench", "swe-bench/swe-bench", "openai/human-eval", "hendrycks/math",
    "facebookresearch/xformers", "dao-ailab/flash-attention", "lm-sys/fastchat",
    "eleutherai/lm-evaluation-harness", "pytorch/vision", "kornia/kornia",
}
POSITIVE_CTX = re.compile(
    r"(code|model|models|implementation|source|checkpoints?|weights|project page|release[ds]?|"
    r"open-?source[d]?|publicly|available|our (?:code|repo))", re.I)
NEGATIVE_CTX = re.compile(
    r"(based on|built (?:up)?on|following|adopt|adapted from|borrow|forked from|uses?\b|"
    r"using|baseline|official implementation of \[|pip install|dependenc|compared (?:with|to))", re.I)


def normalise_github(owner: str, repo: str) -> str | None:
    repo = re.sub(r"\.git$", "", repo).rstrip(".,;:)")
    repo = re.split(r"[/?#]", repo)[0]
    if not repo or owner.lower() in {"orgs", "sponsors", "topics", "features"}:
        return None
    return f"https://github.com/{owner}/{repo}"


def find_github_candidates(doc: "fitz.Document", full_text: str, abstract: str | None,
                           title: str | None, authors: list[str]) -> list[dict]:
    refs_at = None
    m = list(re.finditer(r"\n\s*(?:\d+\.?\s*)?(references|bibliography)\s*\n", full_text, re.I))
    if m:
        refs_at = m[-1].start()

    flat = re.sub(r"\s*\n\s*", " ", full_text)
    cands: dict[str, dict] = {}

    def add(url: str, pos: int, ctx: str, in_link_annot: bool = False):
        c = cands.setdefault(url.lower(), {"url": url, "score": 0.0, "first_pos": pos, "ctx": []})
        c["first_pos"] = min(c["first_pos"], pos)
        c["ctx"].append(ctx)
        if in_link_annot:
            c["score"] += 0.5

    for mm in GH_RE.finditer(full_text):
        url = normalise_github(mm.group(1), mm.group(2))
        if not url:
            continue
        ctx = normalise_ws(full_text[max(0, mm.start() - 160): mm.end() + 60])
        add(url, mm.start(), ctx)
    for page in doc:  # clickable links catch URLs split across lines
        for lk in page.get_links():
            uri = lk.get("uri") or ""
            mm = GH_RE.search(uri)
            if mm:
                url = normalise_github(mm.group(1), mm.group(2))
                if url:
                    add(url, 0, "link-annotation", True)

    abstract_l = (abstract or "").lower()
    title_tokens = set(re.findall(r"[a-z0-9]{3,}", (title or "").lower().replace("-", "")))
    author_tokens = {t.lower() for a in authors for t in a.split() if len(t) > 3}
    for key, c in cands.items():
        owner, repo = c["url"].split("github.com/")[1].split("/")[:2]
        slug = f"{owner}/{repo}".lower()
        s = c["score"]
        if slug in DEPENDENCY_REPOS:
            s -= 20
        if repo.lower() in abstract_l or c["url"].lower().split("//")[1] in abstract_l:
            s += 6
        pos_abs = full_text.lower().find(c["url"].lower().split("//")[1])
        if abstract and pos_abs != -1 and pos_abs < len(abstract) + 1500:
            s += 3                                    # appears in header / first-page footnote
        if refs_at is not None and c["first_pos"] > refs_at:
            s -= 6                                    # only in bibliography
        joined = " ".join(c["ctx"])
        if POSITIVE_CTX.search(joined):
            s += 3
        if NEGATIVE_CTX.search(joined):
            s -= 3
        if len(c["ctx"]) > 1:
            s += min(2, 0.5 * len(c["ctx"]))
        repo_tokens = set(re.findall(r"[a-z0-9]{3,}", repo.lower().replace("-", "")))
        if repo_tokens & title_tokens:
            s += 3
        if owner.lower() in author_tokens or any(owner.lower() in t for t in author_tokens):
            s += 2
        c["score"] = s
    return sorted(cands.values(), key=lambda c: (-c["score"], c["first_pos"]))


def pick_official_repo(cands: list[dict], min_score: float = 3.0) -> tuple[str | None, bool]:
    if not cands or cands[0]["score"] < min_score:
        return None, False
    return cands[0]["url"], True


# --------------------------------------------------------------------------- #
# Models / datasets / claims - heuristics
# --------------------------------------------------------------------------- #
MODEL_PATTERNS = [
    r"GPT-?[0-9][\w.\-]*(?:o|-mini|-turbo)?", r"Claude[- ]?(?:\d(?:\.\d)?[- ]?)?(?:Opus|Sonnet|Haiku)[\w. \-]*?(?=[\s,;)])",
    r"Gemini[- ]?\d(?:\.\d)?[- ](?:Pro|Flash|Ultra)", r"Gemma-?\d*[\w\-]*", r"Qwen-?(?:\d(?:\.\d)?)?[- ]?(?:VL|Coder)?[\w.\-]*",
    r"Llama-?\d?(?:\.\d)?[\w.\-]*", r"DeepSeek-?[A-Za-z]*-?V?\d[\w.\-]*", r"MiniMax-?M[\d.]+", r"GLM-?\d[\w.\-]*",
    r"BitNet(?:[ -]?b1\.58)?[\w.\-]*", r"Mistral[\w.\-]*", r"Phi-?\d[\w.\-]*", r"DUSt3R", r"MASt3R", r"CUT3R",
    r"Fast3R", r"UniMERNet", r"Pix2tex|pix2tex", r"Donut-?\w*", r"CoTracker\d?", r"LocoTrack", r"RoMa", r"LoFTR",
    r"SuperGlue", r"MVSNet", r"MV-DUSt3R", r"FLARE", r"LVSM", r"GS-LRM", r"LGM", r"TAPTR", r"BootsTAPIR", r"PixSfM", r"CasMTR", r"DKM", r"VGGSfM(?: ?v2)?", r"ResNet-?\d+", r"Vary-?VIT-?B", r"PP-HGNetV2(?:-B\d)?", r"GOT-?OCR[\d.]*", r"SWE-agent", r"OpenHands",
    r"Agentless", r"Trae[- ]?[Aa]gent", r"Live-SWE-agent", r"Mini-SWE-agent", r"Moatless[- ]?[Tt]ools", r"AutoCodeRover",
]
MODEL_RE = re.compile(r"\b(" + "|".join(MODEL_PATTERNS) + r")\b")

DATASET_PATTERNS = [
    r"SWE-?[Bb]ench(?:[- ](?:Verified|Lite|Multimodal|Live))?", r"HumanEval\+?", r"MBPP", r"MATH-?500", r"GSM8K", r"MMLU(?:-Pro)?",
    r"GLUE", r"MNLI", r"QNLI", r"SST-?2", r"CNN/?DailyMail|CNNDM", r"ImageNet(?:-1k|-21k)?", r"COCO", r"ScanNet(?:-1500)?",
    r"CO3D(?:v2|Dv2)?", r"Co3Dv2", r"RealEstate10K", r"DTU", r"ETH3D", r"IMC", r"TAP-Vid(?: \w+)?", r"Kubric", r"BlendedMVS|BlendMVS",
    r"DL3DV", r"MegaDepth", r"Hypersim", r"Replica", r"Habitat", r"Objaverse", r"GSO", r"UniMERNet-1M", r"arXiv-4M",
    r"LAION-?\w*", r"The Pile", r"RefinedWeb", r"FALCON corpus", r"WildRGB-?D?", r"Mapillary", r"PointOdyssey", r"Virtual KITTI",
]
DATASET_RE = re.compile(r"\b(" + "|".join(DATASET_PATTERNS) + r")\b")
GENERIC_DS_RE = re.compile(r"\b((?:[A-Z][\w\-]+ ){0,3}[A-Z][\w\-]+) (?:dataset|benchmark|Benchmark|Dataset)\b")
DS_STOP = {"The", "This", "Our", "Each", "A", "An", "Evaluation", "Training", "Test", "Large", "New"}


def _unique(seq):
    seen, out = set(), []
    for x in seq:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _canon_model(m: str) -> str:
    return re.sub(r"\s+", " ", m).strip(" .,-")


MODEL_ALIASES = {"qwen3-cp": "Qwen3-Coder-Plus", "qwen-coder-plus": "Qwen3-Coder-Plus"}
FAMILY_BARE = {"qwen", "qwen3", "qwen2.5", "gemma", "gemma3", "donut", "deepseek", "pp-hgnetv2"}
DATASET_ALIASES = {"sst2": "SST-2", "cnndm": "CNN/DailyMail", "re10k": "RealEstate10K", "re10k dataset": "RealEstate10K",
                   "vid rgb-s dataset": "TAP-Vid", "blendmvs": "BlendedMVS", "swe-bench-verified": "SWE-bench Verified",
                   "dailymail dataset": "CNN/DailyMail", "cnndm dataset": "CNN/DailyMail"}


def _key(s: str) -> str:
    return re.sub(r"[\s\-_/]+", "", s.lower())


def clean_models(models: list[str], title: str | None = None, abstract: str | None = None) -> list[str]:
    """Remove fragments / duplicates / bare family names; use the paper's own acronym for its method."""
    out = []
    for m in models:
        m = MODEL_ALIASES.get(m.lower(), m).strip()
        if (re.search(r"\b(series|technical|report)\b", m, re.I) or m.lower().endswith("-like")
                or re.search(r"\s[a-z]{1,3}$", m) or len(m) < 3):
            continue
        out.append(m)
    if title and abstract:
        head = re.split(r"[:\u2014]\s", title)[0].strip()
        ac = re.search(rf"{re.escape(head)}\s*\(([A-Za-z][A-Za-z0-9\-]+)\)", abstract)
        if ac:
            out = [ac.group(1)] + [m for m in out if _key(m) != _key(head)]
    out = _unique_by_key(out)
    keys = [_key(m) for m in out]
    return [m for m in out if not (_key(m) in {_key(f) for f in FAMILY_BARE} and
                                   any(k != _key(m) and k.startswith(_key(m)) for k in keys))]


def _unique_by_key(seq):
    seen, out = set(), []
    for x in seq:
        k = _key(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def clean_datasets(ds: list[str]) -> list[str]:
    out = []
    for d in ds:
        d = DATASET_ALIASES.get(d.lower(), d).strip()
        if len(d.split()) > 4 or "question-answering" in d.lower():
            continue
        out.append(d)
    out = _unique_by_key(out)
    keys = [_key(d) for d in out]
    return [d for d in out if not (d.lower().endswith((" dataset", " benchmark")) and
                                   any(_key(d.rsplit(" ", 1)[0]) in k and k != _key(d) for k in keys))]


def heuristic_models(text: str, title: str | None, abstract: str | None = None) -> list[str]:
    body = text.split("\nReferences\n")[0] if "\nReferences\n" in text else text
    found = [_canon_model(m.group(0)) for m in MODEL_RE.finditer(body)]
    if title:   # models introduced by the paper: Title-Name + variants such as "PP-FormulaNet-L"
        head = re.split(r"[:\u2014\-]\s", title)[0].strip()
        if head and len(head.split()) <= 2:
            found = re.findall(rf"\b{re.escape(head)}(?:-[A-Z0-9][\w.]*)*\b", body) + found
    return clean_models(_unique([f for f in found if len(f) > 2]), title, abstract)


def heuristic_datasets(text: str) -> list[str]:
    body = text.split("\nReferences\n")[0] if "\nReferences\n" in text else text
    found = [m.group(0) for m in DATASET_RE.finditer(body)]
    for m in GENERIC_DS_RE.finditer(body):
        name = m.group(1).strip()
        if name.split()[0] not in DS_STOP and len(name) > 3:
            found.append(name + (" Benchmark" if "enchmark" in m.group(0) else " Dataset"))
    cleaned = clean_datasets(_unique(found))
    return sorted(cleaned, key=lambda d: -body.count(d.split(" ")[0]))      # most-mentioned first


CLAIM_RE = re.compile(
    r"(?P<value>(?:up to\s+)?\d+(?:\.\d+)?\s?(?:%|x\b|\u00d7|\s?times|\s?pp\b)(?:\s(?:faster|smaller|speedup|reduction|improvement))?)", re.I)
TIME_RE = re.compile(r"(?P<value>(?:under|less than|within)\s+(?:one|a|\d+(?:\.\d+)?)\s+(?:milli)?seconds?)", re.I)
TARGET_RE = re.compile(
    r"(?:than|over|against|compared (?:to|with)|relative to|vs\.?|surpass(?:es|ing)?|outperform(?:s|ing)?|beat(?:s|ing)?)\s+"
    r"(?:those of\s+)?(?:(?:the|prior|previous|existing|state-of-the-art|prominent)\s+)*(?:models\s+)?(?:such as\s+)?"
    r"([A-Z][\w.\-+/]*(?:[ ][A-Z0-9][\w.\-+/]*){0,3})")
BASELINES_RE = re.compile(r"\b(?:over|than|against)\s+((?:all\s+)?(?:the\s+)?baselines)", re.I)
ALT_RE = re.compile(r"outperform(?:s|ing)\s+(alternatives that [^,.]*)", re.I)
FP_RE = re.compile(r"(full-precision (?:counterpart )?(?:models|LLMs?|counterparts?))", re.I)
CONTRAST_RE = re.compile(r"^\s*(?:conversely|meanwhile|similarly|likewise|in contrast)\b", re.I)
CLAUSE_SPLIT = re.compile(r";|,\s*(?:and|while|whereas|but)\s+", re.I)
METRIC_HINTS = [
    (r"pass@1|resolve rate|resolved", "Pass@1"), (r"accuracy|acc\b", "Accuracy"), (r"f1", "F1 Score"),
    (r"speed ?up|faster|inference speed|throughput", "Speedup Factor"), (r"memory|footprint", "Memory Reduction"),
    (r"latency", "Latency Reduction"), (r"bleu", "BLEU"), (r"rouge", "ROUGE"), (r"auc", "AUC"),
    (r"chamfer", "Chamfer Distance"), (r"psnr", "PSNR"), (r"cost", "Cost"),
]
GENERIC_METRIC = (r"improv", "Relative Improvement")
QUALITY_METRICS = {"Accuracy", "Pass@1", "F1 Score", "BLEU", "ROUGE", "AUC", "PSNR", "Chamfer Distance", "Relative Improvement"}


def nearest_metric(text: str, vs: int, ve: int) -> str | None:
    """Specific metric keyword closest to the number; generic 'improvement' only if nothing specific exists."""
    best, best_d = None, 10 ** 9
    for pat, name in METRIC_HINTS:
        for m in re.finditer(pat, text, re.I):
            d = 0 if vs <= m.start() < ve else min(abs(m.start() - ve), abs(vs - m.end()))
            if d < best_d:
                best, best_d = name, d
    if best is None and re.search(GENERIC_METRIC[0], text, re.I):
        return GENERIC_METRIC[1]
    return best


def _find_target(clause: str) -> str | None:
    m = TARGET_RE.search(clause)
    if m:
        return m.group(1).strip()
    m = BASELINES_RE.search(clause)
    if m:
        return m.group(1).strip().capitalize()
    if re.search(r"leaderboard", clause, re.I) and re.search(r"first|top|rank", clause, re.I):
        return "Other leaderboard entries"
    return None


def _find_bench(text: str, datasets: list[str]) -> str | None:
    nt = _key(text)
    hits = [d for d in datasets if _key(d) and _key(d) in nt]
    return max(hits, key=lambda d: len(_key(d))) if hits else None


def heuristic_claims(abstract: str | None, results_text: str, datasets: list[str]) -> list[dict]:
    """Quantitative claims from the abstract; each tied to its own metric, benchmark and baseline."""
    claims, seen, last_target = [], set(), None
    if not abstract:
        return claims
    sents = re.split(r"(?<=[.!?])\s+", abstract)
    for si, sent in enumerate(sents):
        contrast = bool(CONTRAST_RE.match(sent))
        for clause in re.split(CLAUSE_SPLIT, sent):
            vals = [(m, m.group("value")) for m in list(CLAIM_RE.finditer(clause)) + list(TIME_RE.finditer(clause))]
            tgt = _find_target(clause)
            if tgt:
                last_target = tgt
            if not vals:
                continue
            if not tgt:
                alt = ALT_RE.search(sent)
                fp = FP_RE.search(sent)
                if alt:
                    tgt = alt.group(1)[0].upper() + alt.group(1)[1:]
                elif fp:
                    tgt = fp.group(1)[0].upper() + fp.group(1)[1:]
                elif contrast and last_target:
                    tgt = last_target
            for m, raw in vals:
                is_time = TIME_RE.fullmatch(raw) is not None
                metric = "Runtime" if is_time else nearest_metric(clause, m.start(), m.end())
                bench = _find_bench(sent, datasets)
                if not bench and metric in QUALITY_METRICS:
                    bench = next((b for j in (si - 1, si - 2) if j >= 0 for b in [_find_bench(sents[j], datasets)] if b), None)
                    by_target = next((d for d in datasets if tgt and _key(tgt) in _key(d)), None)
                    bench = bench or by_target or (datasets[0] if datasets else None)
                if not bench:
                    bench = {"Speedup Factor": "Inference Speed", "Memory Reduction": "Memory Footprint",
                             "Runtime": "Reconstruction Speed"}.get(metric)
                value = re.sub(r"\s+", " ", raw).strip()
                key = (value.lower(), metric)
                if key in seen:
                    continue
                seen.add(key)
                claims.append({"benchmark": bench, "metric": metric, "value": value, "comparison_target": tgt})
    return claims


TASK_KEYWORDS = {
    "document-understanding": r"document|ocr|formula recognition|layout", "computer-vision": r"image|visual|vision|3d|camera|depth",
    "natural-language-processing": r"language model|\bllms?\b|summariz|natural language|\bnlp\b", "code-generation": r"software engineering|issue resolution|swe-bench|code generation|program repair",
    "agents": r"\bagent", "model-compression": r"quantiz|1\.58|ternary|model compression|\bdistillation\b",
    "3d-reconstruction": r"3d|point map|reconstruction", "tracking": r"tracking",
}
METHOD_KEYWORDS = {
    "deep-learning": r"neural network|\btransformer|deep learning|deep neural", "ocr": r"\bocr\b|formula recognition", "distillation": r"\bdistillation\b",
    "quantization": r"quantiz|1\.58|ternary", "test-time-scaling": r"test-time scaling|ensemble", "reinforcement-learning": r"reinforcement",
    "multi-agent": r"multi-agent", "transformer": r"transformer", "robustness": r"robust",
}


def tag(text: str, table: dict[str, str]) -> list[str]:
    low = text.lower()
    return [k for k, p in table.items() if re.search(p, low)]


# --------------------------------------------------------------------------- #
# LLM-assisted extraction (optional)
# --------------------------------------------------------------------------- #
LLM_PROMPT = """You extract knowledge-graph entities from an AI research paper.
Return ONLY JSON: {{"models": [str], "datasets": [str], "sota_claims": [{{"benchmark": str, "metric": str, "value": str, "comparison_target": str}}]}}
Rules: models = AI models introduced, evaluated or compared in the experiments (not generic words, not libraries).
datasets = training/eval datasets & benchmarks actually used. sota_claims = main quantitative claims from abstract/results
tied to the exact benchmark+metric and the baseline beaten. Never invent values.
PAPER TITLE: {title}
ABSTRACT: {abstract}
BODY (truncated):
{body}"""


def llm_extract(title: str | None, abstract: str | None, body: str) -> dict | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.environ.get("EXTRACT_MODEL", "claude-sonnet-5-5"), max_tokens=3000,
            messages=[{"role": "user", "content": LLM_PROMPT.format(title=title, abstract=abstract, body=body[:60000])}])
        raw = msg.content[0].text
        raw = raw[raw.index("{"): raw.rindex("}") + 1]
        return json.loads(raw)
    except Exception as exc:
        log.warning("LLM extraction failed (%s); falling back to heuristics", exc)
        return None



# --------------------------------------------------------------------------- #
# arXiv API: cross-check / fallback for header fields (optional, needs network)
# --------------------------------------------------------------------------- #
def fetch_arxiv_api(arxiv_id: str, timeout: int = 20) -> dict | None:
    import xml.etree.ElementTree as ET
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "frontieratlas-extractor/1.0"}),
                                    timeout=timeout) as r:
            root = ET.fromstring(r.read())
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = root.find("a:entry", ns)
        if e is None or e.find("a:title", ns) is None:
            return None
        pub = (e.findtext("a:published", "", ns) or "")[:10].replace("-", "")
        return {"title": normalise_ws(e.findtext("a:title", "", ns)),
                "authors": [normalise_ws(a.findtext("a:name", "", ns)) for a in e.findall("a:author", ns)],
                "abstract": normalise_ws(e.findtext("a:summary", "", ns)),
                "publication_date": pub or None}
    except Exception as exc:
        log.info("arXiv API unavailable for %s (%s)", arxiv_id, exc)
        return None


def _sim(a: str | None, b: str | None) -> float:
    import difflib
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() if a and b else 0.0


def reconcile_with_api(rec: dict, api: dict | None) -> None:
    """PDF parse is primary. The arXiv API fills empty fields and replaces parses that clearly disagree."""
    if not api:
        return
    if not rec["title"] or _sim(rec["title"], api["title"]) < 0.85:
        log.info("%s title replaced from arXiv API", rec["arxiv_id"])
        rec["title"] = strip_latex(api["title"])
    api_auth = [clean_author(x) for x in api["authors"] if is_person_name(clean_author(x))]
    if api_auth and [x.lower() for x in rec["authors"]] != [x.lower() for x in api_auth]:
        log.info("%s authors replaced from arXiv API (order/cleanliness)", rec["arxiv_id"])
        rec["authors"] = api_auth
    if not rec["abstract"] or _sim(rec["abstract"], api["abstract"]) < 0.9:
        log.info("%s abstract replaced from arXiv API", rec["arxiv_id"])
        rec["abstract"] = dehyphenate_and_join(strip_latex(api["abstract"]))
    if not rec["publication_date"]:
        rec["publication_date"] = api["publication_date"]

# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def safe(fn, default, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as exc:
        log.warning("%s failed: %s", getattr(fn, "__name__", fn), exc)
        return default


def process_pdf(pdf_path: Path, arxiv_id: str, use_llm: bool = False, use_api: bool = True) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "arxiv_id": arxiv_id, "title": None, "authors": [], "publication_date": None,
        "arxiv_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "arxiv_page_url": f"https://arxiv.org/abs/{arxiv_id}",
        "github_url": None, "is_official_code": False, "tasks": [], "methods": [],
        "models": [], "datasets": [], "sota_claims": [], "abstract": None,
    }
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        log.error("Cannot open %s: %s", pdf_path, exc)
        return rec
    pages_text = [safe(p.get_text, "", "text") for p in doc]
    full_text, first = "\n".join(pages_text), (pages_text[0] if pages_text else "")

    lines = safe(page_lines, [], doc[0]) if len(doc) else []
    title, tidx = safe(extract_title, (None, -1), lines)
    rec["title"] = title
    rec["authors"] = safe(extract_authors, [], lines, tidx) if tidx >= 0 else []
    rec["abstract"] = safe(extract_abstract, None, full_text, first)
    rec["publication_date"] = safe(extract_date, None, doc, arxiv_id)
    if use_api:
        safe(reconcile_with_api, None, rec, safe(fetch_arxiv_api, None, arxiv_id))

    cands = safe(find_github_candidates, [], doc, full_text, rec["abstract"], title, rec["authors"])
    rec["github_url"], rec["is_official_code"] = pick_official_repo(cands)
    log.info("%s github candidates: %s", arxiv_id, [(c["url"], round(c["score"], 1)) for c in cands[:5]])

    llm = safe(llm_extract, None, title, rec["abstract"], full_text) if use_llm else None
    if llm:
        rec["models"] = clean_models(_unique(llm.get("models", [])))
        rec["datasets"] = clean_datasets(_unique(llm.get("datasets", [])))
        rec["sota_claims"] = llm.get("sota_claims", [])
    else:
        rec["models"] = safe(heuristic_models, [], full_text, rec["title"], rec["abstract"])
        rec["datasets"] = safe(heuristic_datasets, [], full_text)
        if "imagenet" not in (rec["abstract"] or "").lower():       # usually backbone pre-training, not an experiment
            rec["datasets"] = [d for d in rec["datasets"] if not d.lower().startswith("imagenet")]
        rec["sota_claims"] = safe(heuristic_claims, [], rec["abstract"], full_text, rec["datasets"])
    ctx = f"{title or ''} {rec['abstract'] or ''}"
    rec["tasks"], rec["methods"] = tag(ctx, TASK_KEYWORDS), tag(ctx, METHOD_KEYWORDS)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", default=None, help="arXiv IDs (default: the 5 test papers)")
    ap.add_argument("--pdf-dir", default="papers", help="folder holding / receiving <id>.pdf files")
    ap.add_argument("--out", default="extracted_papers.json")
    ap.add_argument("--llm", action="store_true", help="LLM-assisted models/datasets/claims (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--no-api", action="store_true", help="do not cross-check header fields with the arXiv API")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    pdf_dir = Path(args.pdf_dir)
    ids = args.ids
    if ids is None:
        on_disk = sorted(p.stem for p in pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        ids = on_disk or DEFAULT_IDS
    results = []
    for aid in ids:
        pdf = get_pdf(aid, pdf_dir)
        if pdf is None:
            results.append(process_empty(aid))
            continue
        results.append(process_pdf(pdf, aid, use_llm=args.llm, use_api=not args.no_api))
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} record(s) to {args.out}")
    return 0


def process_empty(aid: str) -> dict[str, Any]:
    return {"arxiv_id": aid, "title": None, "authors": [], "publication_date": None,
            "arxiv_pdf_url": f"https://arxiv.org/pdf/{aid}.pdf", "arxiv_page_url": f"https://arxiv.org/abs/{aid}",
            "github_url": None, "is_official_code": False, "tasks": [], "methods": [], "models": [],
            "datasets": [], "sota_claims": [], "abstract": None}


if __name__ == "__main__":
    raise SystemExit(main())
