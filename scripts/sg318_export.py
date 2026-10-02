# -*- coding: utf-8 -*-
"""줄 인식 학습용 데이터 내보내기 (PARSeq LMDB).

  python -X utf8 작업도구/스크립트/sg318_export.py [--name v1]

- 학습·검증: 사상계 318면(시험 호 제외). 라벨 = Fable T 재판독을 NDL 세로줄에 정렬한 것.
  사람 검수 판정(검수/*.json)이 있으면 반영한다(아직 없으면 Fable 기본값). 이체자는 정자로 통일.
- 제외: 가로줄, 정렬 품질 불량(길이비 0.8~1.25 밖, 한자 닻 60% 미만), □ 또는 Fable [?]가 든 줄, 61자 이상.
- 시험: 고정 시험세트 60면의 NDL 세로·가로줄 이미지(라벨 없음). 면 단위로 이어 붙여 evaluate_v2로 채점한다.
- 세로줄은 반시계 90° 회전해 가로로 눕힌다(NDLOCR-Lite parseq.py와 같은 처리).
출력: ocr_data/sg318_<name>/{train,val}/data.mdb, test_lines/*.png, test_lines.jsonl, charset.txt, stats.json
"""
import argparse
import hashlib
import io
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import lmdb
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sg318_build import HANJA, OUT as D, ROOT, clean_transcript, line_bounds, load_manifest  # noqa: E402
from sg318_triage import fold  # noqa: E402

BENCH = ROOT / "OCR_사상계/벤치마크_Sol_Astra"


def norm_label(s):
    s = unicodedata.normalize("NFC", s).replace("○", "〇")
    return fold(s)


def crop(img, box, vertical):
    x0, y0, x1, y1 = box
    c = img.crop((max(0, x0 - 2), max(0, y0 - 2), min(img.width, x1 + 2), min(img.height, y1 + 2))).convert("RGB")
    return c.rotate(90, expand=True) if vertical else c


def png(im):
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


def write_lmdb(path, samples):
    path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(path), map_size=4 * 1024 ** 3)
    with env.begin(write=True) as txn:
        for k, (im, lab) in enumerate(samples, 1):
            txn.put(f"image-{k:09d}".encode(), im)
            txn.put(f"label-{k:09d}".encode(), lab.encode("utf-8"))
        txn.put(b"num-samples", str(len(samples)).encode())
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--pages", default=None, help="면 이미지 폴더(기본: 학습데이터_사상계318/pages). 기울기 보정판은 ocr_data/dsk_src_v1/train_pages")
    ap.add_argument("--ndl", default=None, help="NDL json 폴더(기본: 학습데이터_사상계318/ndl)")
    ap.add_argument("--skip-test", action="store_true", help="시험 60면 줄 이미지는 만들지 않는다(이미 있는 export 재사용)")
    a = ap.parse_args()
    PAGES = Path(a.pages) if a.pages else D / "pages"
    NDLD = Path(a.ndl) if a.ndl else D / "ndl"
    X = ROOT / "ocr_data" / f"sg318_{a.name}"  # LMDB가 한글 경로를 못 다뤄 영문 경로에 둔다
    X.mkdir(parents=True, exist_ok=True)
    # 사람 판정(있으면): {항목ID: {v: old|fable|other|variant|illegible, other}}
    decisions = {}
    for f in (D / "검수").glob("sg318_review_*.json"):
        decisions.update(json.loads(f.read_text(encoding="utf-8")).get("decisions", {}))
    queue = {json.loads(l)["id"]: json.loads(l) for l in open(D / "검수" / "review_queue.jsonl", encoding="utf-8")}
    scope = {x["page"] for x in json.loads((D / "검수" / "scope_pages.json").read_text(encoding="utf-8"))}

    split_samples = {"train": [], "val": []}
    stats = Counter()
    rows = []
    for r in load_manifest():
        fp, jp = D / "fable_T" / f"{r['id']}.txt", NDLD / f"{r['id']}.json"
        if not fp.exists() or not jp.exists():
            continue
        F, f_unc = clean_transcript(fp.read_text(encoding="utf-8"))
        # 사람 판정 반영은 기존 전사 좌표 기준 항목이라, 판정이 생기면 별도 단계에서 Fable 문자열에 적용한다(현재 0건).
        lines = [b for blk in json.loads(jp.read_text(encoding="utf-8"))["contents"] for b in blk]
        lb = line_bounds(F, lines)
        if not lb:
            continue
        t2h, owner, H, _, bounds, _ = lb
        img = Image.open(PAGES / f"{r['id']}.png")
        sp = "val" if r["split"] == "validation" else "train"
        for li, s0, s1 in bounds:
            ln = lines[li]
            raw = F[s0:s1 + 1]
            label = norm_label(raw)
            ntext = ln["text"].replace(" ", "")
            vertical = ln.get("isVertical") == "true"
            lh = [k for k in range(s0, s1 + 1) if HANJA.match(F[k])]
            ok = sum(1 for k in lh if t2h[k] is not None and owner[t2h[k]] == li and F[k] == H[t2h[k]])
            ratio = len(raw) / max(1, len(ntext))
            reason = None
            if not vertical:
                reason = "가로줄"
            elif not (0.8 <= ratio <= 1.25) or (lh and ok / len(lh) < 0.6):
                reason = "정렬품질"
            elif "□" in raw or any(s0 <= k <= s1 for k in f_unc):
                reason = "불확실글자"
            elif not 1 <= len(label) <= 60:
                reason = "길이"
            elif r["id"] in scope:
                reason = None  # Fable 라벨은 면 전체를 덮으므로 범위 불일치 면도 사용
            stats[reason or "사용"] += 1
            if reason:
                continue
            xs = [p[0] for p in ln["boundingBox"]]; ys = [p[1] for p in ln["boundingBox"]]
            box = [min(xs), min(ys), max(xs), max(ys)]
            split_samples[sp].append((png(crop(img, box, True)), label))
            rows.append(dict(id=f"{r['id']}_L{li:03d}", page=r["id"], issue=r["issue"], split=sp, box=box, label=label))
    for sp, ss in split_samples.items():
        write_lmdb(X / sp, ss)
    with open(X / "lines.jsonl", "w", encoding="utf-8") as w:
        for x in rows:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    chars = Counter(c for _, lab in split_samples["train"] for c in lab)
    charset = "".join(sorted(chars))
    (X / "charset.txt").write_text(charset, encoding="utf-8")

    if a.skip_test:
        st = dict(filter=dict(stats), train=len(split_samples["train"]), val=len(split_samples["val"]),
                  train_chars=sum(len(l) for _, l in split_samples["train"]), charset_size=len(charset),
                  pages_dir=str(PAGES), ndl_dir=str(NDLD))
        (X / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(st, ensure_ascii=False))
        return
    # 시험 60면: NDL 줄 이미지(세로·가로), 읽기 순서 그대로
    T = X / "test_lines"
    T.mkdir(exist_ok=True)
    test_rows = []
    test_ids = json.loads((BENCH / "04_고정시험세트.json").read_text(encoding="utf-8"))["frozen_test_ids"]
    for sid in test_ids:
        jp = BENCH / "outputs_baseline/NDLOCR" / f"{sid}.json"
        img = Image.open(BENCH / "inputs" / f"{sid}.png")
        lines = [b for blk in json.loads(jp.read_text(encoding="utf-8"))["contents"] for b in blk]
        for li, ln in enumerate(lines):
            xs = [p[0] for p in ln["boundingBox"]]; ys = [p[1] for p in ln["boundingBox"]]
            box = [min(xs), min(ys), max(xs), max(ys)]
            v = ln.get("isVertical") == "true"
            name = f"{sid}_L{li:03d}.png"
            crop(img, box, v).save(T / name)
            test_rows.append(dict(page=sid, order=li, image=f"test_lines/{name}", vertical=v, box=box))
    with open(X / "test_lines.jsonl", "w", encoding="utf-8") as w:
        for x in test_rows:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    test_chars = Counter(c for sid in test_ids for c in unicodedata.normalize("NFC", (BENCH / "references" / f"{sid}.txt").read_text(encoding="utf-8")) if not c.isspace())
    oov = {c: n for c, n in test_chars.items() if norm_label(c) not in chars and c != "[" and c != "?" and c != "]"}
    st = dict(filter=dict(stats), train=len(split_samples["train"]), val=len(split_samples["val"]),
              train_chars=sum(len(l) for _, l in split_samples["train"]), charset_size=len(charset),
              hanja_types=sum(1 for c in charset if HANJA.match(c)),
              test_pages=len(test_ids), test_lines=len(test_rows),
              test_ref_chars=sum(test_chars.values()), test_oov_chars=sum(oov.values()), test_oov_types=len(oov),
              test_oov_top=sorted(oov.items(), key=lambda x: -x[1])[:30],
              max_label_len=max(len(l) for ss in split_samples.values() for _, l in ss))
    (X / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in st.items() if k != "test_oov_top"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
