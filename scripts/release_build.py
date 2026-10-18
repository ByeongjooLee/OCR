# -*- coding: utf-8 -*-
"""공개 저장소(github.com/ByeongjooLee/OCR)용 학습데이터 묶음 만들기.

  python -X utf8 작업도구/스크립트/release_build.py [--src sg318_v2dsk] [--name sasanggye_lines_v1] [--out 공개저장소_OCR]
  python -X utf8 작업도구/스크립트/release_build.py --src sg318_v3dsk --name sasanggye_lines_v2 --layout issue --ocr
  python -X utf8 작업도구/스크립트/release_build.py --src sg_kmmocr_corr_v1 --name sasanggye_kmmocr_corr_v1 --layout issue

- 면 전체 이미지는 넣지 않는다. 잘라낸 줄 이미지와 라벨만 넣는다(축적지침 6절, 2026-10-03 사용자 결정).
- 고정 시험세트 20개 호는 줄 단위로 다시 확인해 하나라도 섞이면 중단한다.
- 줄 이미지는 학습용 회전 전 원래 방향(세로줄은 세로로 선 그림)으로 저장한다.
- --layout flat(v1): images/<split>/<id>.png 한 폴더. --layout issue(10-09 사용자 결정 형식): 호별 폴더 <호>/images/<id>.webp(회색조 WebP 품질 90) + <호>/lines.jsonl.
- --ocr: 줄마다 KMMOCR 인식 결과(주 모델 ocr, 대조 모델 ocr_alt)를 붙인다(GPU). 내보내기에 이미 ocr이 있으면 그대로 쓴다.
- KMMOCR 기사 판독분(sg_kmmocr_corr_*, 줄에 folder 필드)은 면 이미지를 ocr_data/article_pages에서 찾는다.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "OCR_사상계/벤치마크_Sol_Astra"
D318 = ROOT / "학습데이터_사상계318"
ART_PAGES = ROOT / "ocr_data/article_pages"
EXTRA = ("label_prev", "recorr_fixes", "label_raw", "ocr", "ocr_alt", "ocr_model", "minp", "corr")


def add_ocr(rows, crops):
    """KMMOCR 주 모델(PARSeq)·대조 모델(CRNN)로 줄 그림을 읽는다(sg_article_ocr.py와 같은 모델·긴 줄 나눔)."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sg_article_ocr import MAIN, models
    from sg_ocr_predict import split_long
    M = models()
    todo = [i for i, r in enumerate(rows) if not r.get("ocr")]
    for k0 in range(0, len(todo), 2048):
        chunk = todo[k0:k0 + 2048]
        pm, pa, owner = [], [], []
        for i in chunk:
            c = crops[i].convert("RGB")
            c = c.rotate(90, expand=True) if rows[i]["vertical"] else c
            for pc in split_long(c, 700):
                pm.append(M["tfm"](pc)); pa.append(M["tfa"](pc)); owner.append(i)
        tm, ta = {i: "" for i in chunk}, {i: "" for i in chunk}
        with torch.inference_mode():
            for k in range(0, len(owner), 64):
                a, _ = M["main"].tokenizer.decode(M["main"](torch.stack(pm[k:k + 64]).cuda()).softmax(-1))
                b, _ = M["alt"].tokenizer.decode(M["alt"](torch.stack(pa[k:k + 64]).cuda()).softmax(-1))
                for o, s, s2 in zip(owner[k:k + 64], a, b):
                    tm[o] += s; ta[o] += s2
        for i in chunk:
            rows[i].update(ocr=tm[i], ocr_alt=ta[i], ocr_model=f"KMMOCR {Path(MAIN).stem}")
        print(f"  ocr {k0 + len(chunk)}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sg318_v2dsk")
    ap.add_argument("--pages", default=str(ROOT / "ocr_data/dsk_src_v1/train_pages"))
    ap.add_argument("--name", default="sasanggye_lines_v1")
    ap.add_argument("--out", default=str(ROOT / "공개저장소_OCR"))
    ap.add_argument("--layout", choices=["flat", "issue"], default="flat")
    ap.add_argument("--ocr", action="store_true")
    a = ap.parse_args()
    src = ROOT / "ocr_data" / a.src
    pages = Path(a.pages)
    out = Path(a.out) / "data" / a.name
    if a.layout == "issue" and out.exists():
        sys.exit(f"이미 있음(공개 버전은 고치지 않는다): {out}")
    test_ids = json.loads((BENCH / "04_고정시험세트.json").read_text(encoding="utf-8"))["frozen_test_ids"]
    test_issues = sorted({t.split("_")[1] for t in test_ids})
    manifest = {json.loads(l)["id"]: json.loads(l) for l in open(D318 / "manifest.jsonl", encoding="utf-8")}
    angles = json.loads((pages / "angles.json").read_text(encoding="utf-8")) if (pages / "angles.json").exists() else {}

    rows, crops, cache, stats = [], [], {}, Counter()
    for l in open(src / "lines.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["issue"] in test_issues or r["page"] in test_ids:
            sys.exit(f"시험 호가 섞였다: {r['id']}")
        art = "folder" in r   # KMMOCR 기사 판독분
        ip = ART_PAGES / f"sg_{r['issue']}_{r['page'].split('_p')[1]}.png" if art else pages / f"{r['page']}.png"
        if ip not in cache:
            cache.clear()
            cache[ip] = Image.open(ip)
        img = cache[ip]
        x0, y0, x1, y1 = r["box"]
        c = img.crop((max(0, x0 - 2), max(0, y0 - 2), min(img.width, x1 + 2), min(img.height, y1 + 2))).convert("L")
        if art:
            meta = dict(page_no=int(r["page"].split("_p")[1]), article=r["article"], deskew_deg=r.get("deskew_deg", 0.0),
                        vertical=r.get("dir", "v") == "v")
        else:
            m = manifest[r["page"]]
            meta = dict(page_no=m["page"], article=Path(m["transcript_file"]).stem.replace(".txt + ", " + "),
                        deskew_deg=angles.get(r["page"], 0.0), vertical=True)
        row = dict(id=r["id"], split=r["split"], label=r["label"], magazine="사상계", issue_id=r["issue"], page_id=r["page"],
                   page_no=meta["page_no"], article=meta["article"], box=r["box"], deskew_deg=meta["deskew_deg"],
                   vertical=meta["vertical"], grade=r.get("grade", "silver"), rights="unknown")
        row.update({k: r[k] for k in EXTRA if k in r})
        if art and r.get("ocr") and "ocr_model" not in r:
            row["ocr_model"] = "KMMOCR E20avg12_parseq (기사 판독 초벌)"
        rows.append(row); crops.append(c)
        stats[r["split"]] += 1
    if a.ocr:
        add_ocr(rows, crops)
    for row, c in zip(rows, crops):
        if a.layout == "flat":
            rel = f"images/{row['split']}/{row['id']}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            c.save(out / rel, optimize=True)
            row["image"] = rel
        else:
            rel = f"images/{row['id']}.webp"
            (out / row["issue_id"] / "images").mkdir(parents=True, exist_ok=True)
            c.save(out / row["issue_id"] / rel, "WEBP", quality=90, method=6)
            row["image"] = rel
    order = ["id", "split", "image", "label"]
    rows = [{k: x[k] for k in order} | {k: v for k, v in x.items() if k not in order} for x in rows]
    if a.layout == "flat":
        groups = {None: rows}
    else:
        groups = {}
        for x in rows:
            groups.setdefault(x["issue_id"], []).append(x)
    shas = {}
    for iss, rs in groups.items():
        p = out / "lines.jsonl" if iss is None else out / iss / "lines.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as w:
            for x in rs:
                w.write(json.dumps(x, ensure_ascii=False) + "\n")
        shas[iss or "lines.jsonl"] = hashlib.sha256(p.read_bytes()).hexdigest()
    chars = Counter(c for x in rows if x["split"] == "train" for c in x["label"])
    (out / "charset.txt").write_text("".join(sorted(chars)), encoding="utf-8")
    src_stats = json.loads((src / "stats.json").read_text(encoding="utf-8")) if (src / "stats.json").exists() else {}
    st = dict(
        name=a.name, source_export=f"ocr_data/{a.src}", layout=a.layout, lines=dict(stats),
        pages=len({x["page_id"] for x in rows}), issues=len({x["issue_id"] for x in rows}),
        train_chars=sum(len(x["label"]) for x in rows if x["split"] == "train"),
        charset_size=len(chars),
        hanja_types=sum(1 for c in chars if re.match(r"[㐀-䶿一-鿿豈-﫿]", c)),
        grade=dict(Counter(x["grade"] for x in rows)),
        ocr_model=rows[0].get("ocr_model") if rows and rows[0].get("ocr_model") else None,
        heldout_test_issues_excluded=test_issues,
        source_filter=src_stats.get("filter"),
    )
    if a.layout == "flat":
        st["lines_jsonl_sha256"] = shas["lines.jsonl"]
    else:
        st["lines_jsonl_sha256_by_issue"] = shas
    (out / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    splits = Path(a.out) / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "heldout_test_issues.json").write_text(json.dumps(dict(
        note="고정 시험세트가 든 호. 이 호들은 학습·언어모델·자가학습에 쓰지 않는다. 시험 면의 이미지와 라벨은 논문 게재 전까지 공개하지 않는다.",
        issues=test_issues, pages=len(test_ids)), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in st.items() if not k.startswith("lines_jsonl")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
