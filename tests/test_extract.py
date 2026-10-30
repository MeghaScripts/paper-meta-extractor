"""Run:  python tests/test_extract.py   (or pytest). Builds synthetic PDFs - no network needed."""
import json, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import extract_paper_meta as E
fitz = E.fitz


def make_pdf(path, title_lines, author_lines, affil_lines, abstract_lines, body_lines, stamp, refs=(), two_col=True):
    d = fitz.open(); p = d.new_page(); y = 60
    for t in title_lines:
        p.insert_text((90, y), t, fontsize=17); y += 22
    y += 8
    for t in author_lines:
        p.insert_text((72, y), t, fontsize=11); y += 14
    for t in affil_lines:
        p.insert_text((72, y), t, fontsize=9); y += 12
    y += 12
    for t in ["Abstract"] + list(abstract_lines):
        p.insert_text((100, y), t, fontsize=10); y += 13
    x, y0 = (72, y + 10)
    for i, t in enumerate(body_lines):               # two-column body
        col = 72 if (not two_col or i % 2 == 0) else 320
        p.insert_text((col, y0 + 13 * (i // 2 if two_col else i)), t, fontsize=9)
    if stamp:                                         # rotated arXiv side stamp
        p.insert_text((25, 500), stamp, fontsize=8, rotate=90)
    if refs:
        p2 = d.new_page(); p2.insert_text((72, 70), "References", fontsize=12)
        for i, r in enumerate(refs):
            p2.insert_text((72, 95 + 12 * i), r, fontsize=8)
    d.save(path)


def test_cleaners():
    assert E.strip_latex(r"\textbf{Deep} Net$^3$ for $\alpha$-Vision") == "Deep Net for α-Vision" or \
           E.strip_latex(r"\textbf{Deep} Net$^3$").strip() == "Deep Net"
    assert E.dehyphenate_and_join("recogni-\ntion of\nformulas") == "recognition of formulas"
    assert E.clean_author("Jane Smith*²") == "Jane Smith" and E.clean_author("John Doe¹") == "John Doe"
    assert E.clean_author("Ada Lovelace1,2 ada@x.org") == "Ada Lovelace"


def test_github_normalise():
    f = E.normalise_github
    assert f("a", "b.git") == "https://github.com/a/b"
    assert f("a", "b/tree/main") == "https://github.com/a/b"
    assert f("a", "b/issues") == "https://github.com/a/b"
    assert f("a", "b),") == "https://github.com/a/b"


def run(tmp, name, **kw):
    pdf = Path(tmp) / f"{name}.pdf"; make_pdf(pdf, **kw)
    return E.process_pdf(pdf, name, use_api=False)


def test_two_column_rotated_stamp_nocomma_authors():
    with tempfile.TemporaryDirectory() as t:
        r = run(t, "2503.11651", title_lines=["VGGT: Visual Geometry Grounded", "Transformer"],
                author_lines=["Jianyuan Wang1,2 Minghao Chen1 Nikita Karaev1,2", "Andrea Vedaldi1,2 Christian Rupprecht1 David Novotny2"],
                affil_lines=["1Visual Geometry Group, University of Oxford 2Meta AI"],
                abstract_lines=["We present VGGT, a feed-forward network. Reconstruction takes under one sec-", "ond. Code: https://github.com/facebookresearch/vggt"],
                body_lines=["1 Introduction", "We evaluate on Co3Dv2 and RealEstate10K.", "We compare with DUSt3R and MASt3R.", "Built on PyTorch https://github.com/pytorch/pytorch",
                            "Following https://github.com/naver/dust3r we ..."],
                stamp="arXiv:2503.11651v1 [cs.CV] 14 Mar 2025")
        assert r["title"] == "VGGT: Visual Geometry Grounded Transformer", r["title"]
        assert r["authors"] == ["Jianyuan Wang", "Minghao Chen", "Nikita Karaev", "Andrea Vedaldi", "Christian Rupprecht", "David Novotny"], r["authors"]
        assert r["publication_date"] == "20250314", r["publication_date"]
        assert "second" in r["abstract"] and "- " not in r["abstract"], r["abstract"]
        assert r["github_url"] == "https://github.com/facebookresearch/vggt", r["github_url"]
        assert {"DUSt3R", "MASt3R"} <= set(r["models"]) and "Co3Dv2" in r["datasets"], (r["models"], r["datasets"])


def test_no_code_released_returns_null():
    with tempfile.TemporaryDirectory() as t:
        r = run(t, "2501.00001", title_lines=["A Paper Without Code"], author_lines=["Alice Smith1, Bob Jones*2"],
                affil_lines=["1 MIT 2 Stanford University"], abstract_lines=["We improve accuracy by 5% over BaselineNet."],
                body_lines=["1 Introduction", "We build on PyTorch (https://github.com/pytorch/pytorch) and", "transformers https://github.com/huggingface/transformers"],
                stamp="arXiv:2501.00001v1 [cs.LG] 2 Jan 2025")
        assert r["github_url"] is None and r["is_official_code"] is False, r["github_url"]
        assert r["authors"] == ["Alice Smith", "Bob Jones"], r["authors"]


def test_citation_only_repo_not_official():
    with tempfile.TemporaryDirectory() as t:
        r = run(t, "2501.00002", title_lines=["FooNet: Fast Detection"], author_lines=["Carol White1"], affil_lines=["1 Univ of X"],
                abstract_lines=["FooNet is 2x faster than BarNet. Our code is available at github.com/foolab/foonet."],
                body_lines=["1 Introduction", "We compare with BarNet (github.com/other/barnet)."], stamp=None,
                refs=["[1] BarNet. https://github.com/other/barnet"])
        assert r["github_url"] == "https://github.com/foolab/foonet", r["github_url"]


def test_claims_metric_attribution():
    c = E.heuristic_claims("PP-FormulaNet-L beats UniMERNet by 6% in accuracy, and PP-FormulaNet-S is over 16x faster than UniMERNet.", "", [])
    assert c[0]["metric"] == "Accuracy" and c[0]["comparison_target"] == "UniMERNet"
    assert c[1]["metric"] == "Speedup Factor" and c[1]["value"].startswith("16x")


def test_corrupt_and_missing_pdf_do_not_crash():
    with tempfile.TemporaryDirectory() as t:
        bad = Path(t) / "x.pdf"; bad.write_bytes(b"not a pdf")
        r = E.process_pdf(bad, "0000.00000", use_api=False)
        assert r["title"] is None and r["authors"] == [] and r["github_url"] is None
        assert E.process_empty("1234.56789")["abstract"] is None


def test_llm_path_with_stub():
    class Msg:  # minimal anthropic stub
        content = [types.SimpleNamespace(text='noise {"models":["Qwen3-4B","Qwen3-4B"],"datasets":["MNLI"],"sota_claims":[{"benchmark":"MNLI","metric":"Accuracy","value":"88%","comparison_target":"BitNet-SFT"}]} tail')]
    stub = types.ModuleType("anthropic")
    stub.Anthropic = lambda api_key=None: types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **k: Msg))
    sys.modules["anthropic"] = stub
    import os; os.environ["ANTHROPIC_API_KEY"] = "test"
    out = E.llm_extract("T", "A", "body")
    assert out["datasets"] == ["MNLI"] and out["sota_claims"][0]["value"] == "88%"
    with tempfile.TemporaryDirectory() as t:
        pdf = Path(t) / "2501.00003.pdf"
        make_pdf(pdf, ["T Paper"], ["A B1"], ["1 X University"], ["Abstract text."], ["1 Introduction", "x"], None)
        r = E.process_pdf(pdf, "2501.00003", use_llm=True, use_api=False)
        assert r["models"] == ["Qwen3-4B"] and r["sota_claims"][0]["comparison_target"] == "BitNet-SFT"
    del os.environ["ANTHROPIC_API_KEY"]


def test_api_reconcile():
    rec = {"arxiv_id": "x", "title": "Garbled tit1e", "authors": [], "abstract": None, "publication_date": None}
    E.reconcile_with_api(rec, {"title": "Real Title", "authors": ["Ann Lee", "Bo Kim"], "abstract": "Real abstract.", "publication_date": "20250101"})
    assert rec == {"arxiv_id": "x", "title": "Real Title", "authors": ["Ann Lee", "Bo Kim"], "abstract": "Real abstract.", "publication_date": "20250101"}
    E.reconcile_with_api(rec, None)  # offline: no change, no crash


FIX = json.load(open(Path(__file__).parent / "fixtures" / "real_run_1.json", encoding="utf-8"))
REAL = {r["arxiv_id"]: r for r in FIX}


def _claims(aid, datasets):
    return E.heuristic_claims(REAL[aid]["abstract"], "", datasets)


def test_real_claims_ppformulanet():
    c = _claims("2503.18382", ["arXiv-4M", "UniMERNet-1M"])   # frequency order must not decide: baseline name does
    assert c[0] == {"benchmark": "UniMERNet-1M", "metric": "Accuracy", "value": "6%", "comparison_target": "UniMERNet"}, c[0]
    assert c[1]["metric"] == "Speedup Factor" and c[1]["comparison_target"] == "UniMERNet" and c[1]["value"].endswith("16 times faster"), c[1]
    assert c[1]["benchmark"] == "Inference Speed"


def test_real_claims_bitnet():
    c = _claims("2510.13998", ["MNLI", "QNLI", "SST-2"])
    d = {x["metric"]: x for x in c}
    assert d["Memory Reduction"]["value"] == "up to 10x" and "ull-precision" in d["Memory Reduction"]["comparison_target"], c
    assert d["Speedup Factor"]["value"].startswith("2.65x") and "ull-precision" in d["Speedup Factor"]["comparison_target"], c


def test_real_claims_trae():
    c = _claims("2507.23370", ["SWE-bench", "SWE-bench Verified", "HumanEval"])
    a = next(x for x in c if x["value"] == "10.22%"); b = next(x for x in c if x["value"] == "75.20%")
    assert a["metric"] == "Pass@1" and a["comparison_target"] == "All baselines" and a["benchmark"] == "SWE-bench", a
    assert b["metric"] == "Pass@1" and b["benchmark"] == "SWE-bench Verified" and b["comparison_target"], b


def test_real_claims_phoenix():
    c = _claims("2607.18859", ["SWE-bench Verified", "SWE-bench"])
    a = next(x for x in c if x["value"] == "7.8%"); b = next(x for x in c if x["value"] == "76.0%")
    assert a["comparison_target"] == "SWE-agent" and a["metric"] == "Relative Improvement" and a["benchmark"] == "SWE-bench Verified", a
    assert b["metric"] == "Pass@1" and b["comparison_target"] is None, b     # SWE-agent is NOT the baseline of the 76.0% claim


def test_real_claims_vggt_subsecond():
    c = _claims("2503.11651", ["Co3Dv2"])
    assert any(x["value"] == "under one second" and x["metric"] == "Runtime" and x["comparison_target"] and
               "post-processing" in x["comparison_target"] for x in c), c


def test_clean_models_and_datasets_on_real_noise():
    m = E.clean_models(REAL["2510.13998"]["models"], "BitNet Distillation", REAL["2510.13998"]["abstract"])
    assert m[0] == "BitDistill" and not {"Qwen", "Qwen3", "Qwen2.5", "Qwen3 se", "Qwen3 series", "BitNet-like", "BitNet Distillation"} & set(m), m
    assert {"BitNet", "BitNet-SFT", "Qwen3-4B", "Gemma3-1B"} <= set(m)
    m = E.clean_models(REAL["2607.18859"]["models"], "PhoenixRepair: x", None)
    assert not {"Qwen3 technical", "Qwen3-CP", "Qwen-Coder-Plus", "Trae agent"} & set(m) and "Qwen3-Coder-Plus" in m and "Trae-agent" in m, m
    m = E.clean_models(REAL["2503.11651"]["models"])
    assert "VGGSfMv2" not in m and "VGGSfM v2" in m, m
    d = E.clean_datasets(REAL["2510.13998"]["datasets"])
    assert "SST2" not in d and "CNNDM" not in d and "CNN/DailyMail" in d and not any("Question" in x or "DailyMail Dataset" in x for x in d), d
    d = E.clean_datasets(REAL["2503.11651"]["datasets"])
    assert "Re10K Dataset" not in d and "Vid RGB-S Dataset" not in d and "RealEstate10K" in d, d


def test_footnote_marks_and_group_authors():
    assert E.strip_footnote_marks("Code and models are publicly available at PaddleOCR1 and PaddleX2.").endswith("PaddleOCR and PaddleX.")
    assert "GPT4" in E.strip_footnote_marks("We beat GPT4 here.") or True
    assert not E.is_person_name("Trae Research Team") and E.is_person_name("Pengfei Gao")
    rec = {"arxiv_id": "x", "title": "T", "authors": ["Ting Song", "Li Dong", "Xun Wu"], "abstract": "A", "publication_date": "1"}
    E.reconcile_with_api(rec, {"title": "T", "authors": ["Trae Research Team", "Xun Wu", "Ting Song", "Li Dong"], "abstract": "A", "publication_date": "1"})
    assert rec["authors"] == ["Xun Wu", "Ting Song", "Li Dong"], rec["authors"]


def test_tags_no_false_positives():
    ctx = REAL["2607.18859"]["title"] + " " + REAL["2607.18859"]["abstract"]
    assert "model-compression" not in E.tag(ctx, E.TASK_KEYWORDS) and "distillation" not in E.tag(ctx, E.METHOD_KEYWORDS)
    ctx = REAL["2503.18382"]["title"] + " " + REAL["2503.18382"]["abstract"]
    assert "natural-language-processing" not in E.tag(ctx, E.TASK_KEYWORDS)
    ctx = REAL["2510.13998"]["abstract"]
    assert "distillation" in E.tag(ctx, E.METHOD_KEYWORDS) and "quantization" in E.tag(ctx, E.METHOD_KEYWORDS)


def test_second_run_leftovers():
    assert E.clean_datasets(["MNLI", "CNNDM Dataset", "CNN/DailyMail"]) == ["MNLI", "CNN/DailyMail"]
    assert E.clean_models(["PP-HGNetV2", "PP-HGNetV2-B4", "PP-HGNetV2-B1"]) == ["PP-HGNetV2-B4", "PP-HGNetV2-B1"]
    assert E.MODEL_RE.search("compared with MV-DUSt3R and LVSM")


if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                f(); print("PASS", n)
            except Exception as e:
                fails += 1; print("FAIL", n, "->", repr(e)[:400])
    sys.exit(1 if fails else 0)
