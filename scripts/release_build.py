# -*- coding: utf-8 -*-
"""공개 저장소(github.com/ByeongjooLee/OCR)용 학습데이터 묶음 만들기.

  python -X utf8 작업도구/스크립트/release_build.py [--src sg318_v2dsk] [--name sasanggye_lines_v1] [--out 공개저장소_OCR]

- 면 전체 이미지는 넣지 않는다. 잘라낸 줄 이미지와 라벨만 넣는다(축적지침 6절, 2026-10-03 사용자 결정).
- 고정 시험세트 20개 호는 줄 단위로 다시 확인해 하나라도 섞이면 중단한다.
- 줄 이미지는 학습용 회전 전 원래 방향(세로줄은 세로로 선 그림)으로 저장한다.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "OCR_사상계/벤치마크_Sol_Astra"
D318 = ROOT / "학습데이터_사상계318"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sg318_v2dsk")
    ap.add_argument("--pages", default=str(ROOT / "ocr_data/dsk_src_v1/train_pages"))
    ap.add_argument("--name", default="sasanggye_lines_v1")
    ap.add_argument("--out", default=str(ROOT / "공개저장소_OCR"))
    a = ap.parse_args()
    src = ROOT / "ocr_data" / a.src
    pages = Path(a.pages)
    out = Path(a.out) / "data" / a.name
    test_ids = json.loads((BENCH / "04_고정시험세트.json").read_text(encoding="utf-8"))["frozen_test_ids"]
    test_issues = sorted({t.split("_")[1] for t in test_ids})
    manifest = {json.loads(l)["id"]: json.loads(l) for l in open(D318 / "manifest.jsonl", encoding="utf-8")}
    angles = json.loads((pages / "angles.json").read_text(encoding="utf-8"))

    rows, cache, stats = [], {}, Counter()
    for l in open(src / "lines.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["issue"] in test_issues or r["page"] in test_ids:
            sys.exit(f"시험 호가 섞였다: {r['id']}")
        if r["page"] not in cache:
            cache.clear()
            cache[r["page"]] = Image.open(pages / f"{r['page']}.png")
        img = cache[r["page"]]
        x0, y0, x1, y1 = r["box"]
        c = img.crop((max(0, x0 - 2), max(0, y0 - 2), min(img.width, x1 + 2), min(img.height, y1 + 2))).convert("L")
        rel = f"images/{r['split']}/{r['id']}.png"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        c.save(out / rel, optimize=True)
        m = manifest[r["page"]]
        rows.append(dict(
            id=r["id"], split=r["split"], image=rel, label=r["label"],
            magazine="사상계", issue_id=r["issue"], page_id=r["page"], page_no=m["page"],
            article=Path(m["transcript_file"]).stem,
            box=r["box"], deskew_deg=angles.get(r["page"], 0.0), vertical=True,
            grade="silver", rights="unknown",
        ))
        stats[r["split"]] += 1
    with open(out / "lines.jsonl", "w", encoding="utf-8") as w:
        for x in rows:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    chars = Counter(c for x in rows if x["split"] == "train" for c in x["label"])
    (out / "charset.txt").write_text("".join(sorted(chars)), encoding="utf-8")
    st = dict(
        name=a.name, source_export=f"ocr_data/{a.src}", lines=dict(stats),
        pages=len({x["page_id"] for x in rows}), issues=len({x["issue_id"] for x in rows}),
        train_chars=sum(len(x["label"]) for x in rows if x["split"] == "train"),
        charset_size=len(chars),
        hanja_types=sum(1 for c in chars if "一" <= c <= "鿿" or "豈" <= c <= "﫿" or "㐀" <= c <= "䶿"),
        grade=dict(Counter(x["grade"] for x in rows)),
        heldout_test_issues_excluded=test_issues,
        lines_jsonl_sha256=hashlib.sha256((out / "lines.jsonl").read_bytes()).hexdigest(),
    )
    (out / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    splits = Path(a.out) / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "heldout_test_issues.json").write_text(json.dumps(dict(
        note="고정 시험세트가 든 호. 이 호들은 학습·언어모델·자가학습에 쓰지 않는다. 시험 면의 이미지와 라벨은 논문 게재 전까지 공개하지 않는다.",
        issues=test_issues, pages=len(test_ids)), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
