# -*- coding: utf-8 -*-
"""사상계 318면 줄 라벨 — KMMOCR 인식 결과로 줄 경계를 다시 잡은 판 (2026-10-18, 라벨 품질 검토 뒤).

  작업도구/ocr_train/venv/Scripts/python.exe -X utf8 작업도구/스크립트/sg318_export_ocralign.py [--name v5dsk] [--thr 0.8]

왜(10-18 품질 검토에서 확인):
- v1~v2dsk의 줄 경계는 NDL 판독과 한자 닻으로 정했다. 한자가 적은 면(시·한글 위주)에서 라벨이 몇 글자~한 줄 통째로 밀린 줄이 많았다
  (라벨과 KMMOCR 인식의 유사도 0.8 미만 402줄, 표본 20줄 중 19줄이 경계 밀림).
- 재교정 판정을 Fable 라벨에 얹은 판(v3dsk·v4dsk)은 Fable과 제 판정이 갈린 자리에서 판정이 틀린 예(철저한→절저한, 文學觀→文學視)가 나왔다.
방법:
- 글자는 Fable T 재판독 그대로(silver). 줄 경계는 면의 NDL 줄마다 KMMOCR(E20avg12) 인식 결과를 이어 붙인 문자열과 Fable 면 문자열을
  글자 단위로 맞춰 정한다(이체자는 정자로 접어 비교). 인식 결과는 경계 잡기에만 쓰고 라벨 글자로 쓰지 않는다.
- 줄 경계 자리에서 Fable에만 있는 글자(어느 줄인지 모름)가 있으면 양쪽 줄을 뺀다.
- 라벨과 그 줄 인식 결과의 유사도가 --thr 미만이면 뺀다(경계 의심·다른 기사 영역).
- 재교정 판정이 Fable과 갈린 자리(Fable=기존 전사인데 판정은 OCR 등)와 불명 자리가 든 줄은 뺀다(사람 검수 후보). 판정과 Fable이 같은 자리는 그대로.
- 가로줄, [?]·□, 61자 이상은 sg318_export.py와 같이 뺀다.
출력: ocr_data/sg318_<name>/{train,val}/data.mdb, lines.jsonl(label·label_prev(v2dsk)·ocr·ocr_alt·sim), excluded.jsonl, stats.json
인식 결과 캐시: ocr_data/sg318_lineocr_E20avg12.json (면별 NDL 줄 순서)
"""
import argparse
import difflib
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sg318_build import OUT as D, ROOT, clean_transcript, load_manifest  # noqa: E402
from sg318_export import crop, norm_label, png, write_lmdb  # noqa: E402
from sg318_export_recorr import BENCH, load_decisions, overlay  # noqa: E402

CACHE = ROOT / "ocr_data/sg318_lineocr_E20avg12.json"


def line_ocr(pages, ndl_dir, page_dir):
    """면마다 NDL 줄(contents 순서)을 KMMOCR 주·대조 모델로 읽는다. 캐시가 있으면 그대로."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [p for p in pages if p not in cache and (ndl_dir / f"{p}.json").exists()]
    if todo:
        import torch
        from sg_article_ocr import models
        from sg_ocr_predict import split_long
        M = models()
        for n, pid in enumerate(todo, 1):
            lines = [b for blk in json.loads((ndl_dir / f"{pid}.json").read_text(encoding="utf-8"))["contents"] for b in blk]
            img = Image.open(page_dir / f"{pid}.png")
            pm, pa, owner = [], [], []
            for i, ln in enumerate(lines):
                xs = [q[0] for q in ln["boundingBox"]]; ys = [q[1] for q in ln["boundingBox"]]
                c = crop(img, [min(xs), min(ys), max(xs), max(ys)], ln.get("isVertical") == "true")
                for pc in split_long(c, 700):
                    pm.append(M["tfm"](pc)); pa.append(M["tfa"](pc)); owner.append(i)
            tm, ta = [""] * len(lines), [""] * len(lines)
            with torch.inference_mode():
                for k in range(0, len(owner), 64):
                    a, _ = M["main"].tokenizer.decode(M["main"](torch.stack(pm[k:k + 64]).cuda()).softmax(-1))
                    b, _ = M["alt"].tokenizer.decode(M["alt"](torch.stack(pa[k:k + 64]).cuda()).softmax(-1))
                    for o, s, s2 in zip(owner[k:k + 64], a, b):
                        tm[o] += s; ta[o] += s2
            cache[pid] = dict(main=tm, alt=ta)
            if n % 20 == 0 or n == len(todo):
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"  인식 {n}/{len(todo)}", flush=True)
    return cache


def fold_chars(s):
    return [norm_label(c) or c for c in s]


def sim(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return 2 * sum(m.size for m in sm.get_matching_blocks()) / max(1, len(a) + len(b))


def assign(F, texts):
    """Fable 면 문자열 F의 글자마다 줄 번호(없으면 None)와, 경계가 모호한 줄 집합을 낸다."""
    H, owner = [], []
    for li, t in enumerate(texts):
        for ch in t.replace(" ", ""):
            H.append(ch); owner.append(li)
    fa, ha = fold_chars(F), fold_chars("".join(H))
    who = [None] * len(F)
    amb = set()
    sm = difflib.SequenceMatcher(None, fa, ha, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                who[i1 + k] = owner[j1 + k]
        elif op == "replace":
            for k in range(i1, i2):
                who[k] = owner[j1 + (k - i1) * (j2 - j1) // (i2 - i1)]
            ls = {owner[j] for j in range(j1, j2)}
            if len(ls) > 1:   # 대체 구간이 줄 경계를 넘으면 비례 배분이 틀릴 수 있다
                amb |= ls
        elif op == "delete":
            if 0 < j1 < len(H) and owner[j1 - 1] != owner[j1]:
                amb |= {owner[j1 - 1], owner[j1]}   # 줄 경계에 Fable에만 있는 글자 → 어느 줄인지 모름
                for k in range(i1, i2):
                    who[k] = owner[j1 - 1]
            elif 0 < j1 <= len(H):
                for k in range(i1, i2):
                    who[k] = owner[j1 - 1]
            # j1 == 0: 첫 줄 앞(줄 검출이 없는 영역) → 배정 안 함
    return who, amb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v5dsk")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--pages", default=str(ROOT / "ocr_data/dsk_src_v1/train_pages"))
    ap.add_argument("--ndl", default=str(ROOT / "ocr_data/dsk_src_v1/train_ndl"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    PAGES, NDLD = Path(a.pages), Path(a.ndl)
    X = ROOT / "ocr_data" / f"sg318_{a.name}"
    if not a.dry and X.exists():
        sys.exit(f"이미 있음(덮어쓰지 않음): {X}")
    test_ids = json.loads((BENCH / "04_고정시험세트.json").read_text(encoding="utf-8"))["frozen_test_ids"]
    test_issues = {t.split("_")[1] for t in test_ids}
    manifest = load_manifest()
    if any(r["issue"] in test_issues for r in manifest):
        sys.exit("시험 호가 manifest에 있다")
    ocr = line_ocr([r["id"] for r in manifest], NDLD, PAGES)
    prev = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(ROOT / "ocr_data/sg318_v2dsk/lines.jsonl", encoding="utf-8")}
    decs = load_decisions()
    by_key = {}
    for d in decs:
        by_key.setdefault((d["file"], d["page"]), []).append(d)

    samples, rows, excl, stats, ostat = {"train": [], "val": []}, [], [], Counter(), Counter()
    for r in manifest:
        fp, jp = D / "fable_T" / f"{r['id']}.txt", NDLD / f"{r['id']}.json"
        if not fp.exists() or not jp.exists() or r["id"] not in ocr:
            continue
        F, unc = clean_transcript(fp.read_text(encoding="utf-8"))
        pdecs = [d for f in r["transcript_file"].split(" + ") for d in by_key.get((f, f"{r['page']:04d}"), [])]
        edits, log = overlay(F, unc, pdecs)
        bad = set()   # 판정 충돌·불명 자리(F 위치)
        for fa, fb, rep, kind in edits:
            bad |= set(range(fa, max(fb, fa + 1)))
        for x in log:
            act = {"replace": "conflict", "exclude_3way": "conflict"}.get(x.get("action"), x.get("action", "-"))
            ostat[(x["kind"], act)] += 1
        lines = [b for blk in json.loads(jp.read_text(encoding="utf-8"))["contents"] for b in blk]
        main_t, alt_t = ocr[r["id"]]["main"], ocr[r["id"]]["alt"]
        who, amb = assign(F, main_t)
        span = {}
        for k, li in enumerate(who):
            if li is not None:
                span.setdefault(li, []).append(k)
        img = None
        sp = "val" if r["split"] == "validation" else "train"
        for li, ln in enumerate(lines):
            ks = span.get(li)
            if not ks:
                continue
            raw = "".join(F[k] for k in ks)
            label = norm_label(raw)
            o = main_t[li].replace(" ", "")
            s = sim(fold_chars(raw), fold_chars(o))
            lid = f"{r['id']}_L{li:03d}"
            reason = None
            if ln.get("isVertical") != "true":
                reason = "가로줄"
            elif "□" in raw or any(k in unc for k in ks):
                reason = "불확실글자"
            elif any(k in bad for k in ks):
                reason = "재교정충돌·불명"
            elif li in amb:
                reason = "경계모호"
            elif s < a.thr:
                reason = "인식과다름"
            elif not 1 <= len(label) <= 60:
                reason = "길이"
            elif ks[-1] - ks[0] + 1 != len(ks):
                reason = "글자불연속"
            stats[reason or "사용"] += 1
            xs = [q[0] for q in ln["boundingBox"]]; ys = [q[1] for q in ln["boundingBox"]]
            box = [min(xs), min(ys), max(xs), max(ys)]
            if reason:
                excl.append(dict(id=lid, reason=reason, label=label, ocr=o, sim=round(s, 3)))
                continue
            if img is None:
                img = Image.open(PAGES / f"{r['id']}.png")
            p = prev.get(lid)
            stats["사용_v2dsk와같음" if p == label else ("사용_v2dsk와다름" if p is not None else "사용_v2dsk에없던줄")] += 1
            samples[sp].append((png(crop(img, box, True)), label))
            rows.append(dict(id=lid, page=r["id"], issue=r["issue"], split=sp, box=box, label=label, label_prev=p,
                             ocr=main_t[li], ocr_alt=alt_t[li], ocr_model="KMMOCR E20avg12_parseq", sim=round(s, 3),
                             grade="silver"))
    print("판정:", dict(ostat))
    print("줄:", dict(stats))
    if a.dry:
        return
    for s_, ss in samples.items():
        write_lmdb(X / s_, ss)
    for name, data in (("lines.jsonl", rows), ("excluded.jsonl", excl)):
        with open(X / name, "w", encoding="utf-8", newline="\n") as w:
            for x in data:
                w.write(json.dumps(x, ensure_ascii=False) + "\n")
    chars = Counter(c for _, lab in samples["train"] for c in lab)
    charset = "".join(sorted(chars))
    (X / "charset.txt").write_text(charset, encoding="utf-8")
    from sg318_build import HANJA
    st = dict(
        name=f"sg318_{a.name}", created="2026-10-18", script="작업도구/스크립트/sg318_export_ocralign.py",
        label_source="fable_T(글자) + KMMOCR E20avg12 인식으로 줄 경계", sim_threshold=a.thr, base_version="sg318_v2dsk",
        note="10-18 품질 검토로 v3dsk·v4dsk(재교정 얹기) 폐기. 재교정 판정은 Fable과 같은 자리만 확인용으로 쓰고, 갈린 자리는 줄을 뺌",
        filter={k: v for k, v in stats.items() if not k.startswith("사용_")},
        vs_v2dsk={k[3:]: v for k, v in stats.items() if k.startswith("사용_")},
        recorr={f"{k[0]}|{k[1]}": v for k, v in ostat.items()},
        train=len(samples["train"]), val=len(samples["val"]),
        train_chars=sum(len(l) for _, l in samples["train"]), charset_size=len(charset),
        hanja_types=sum(1 for c in charset if HANJA.match(c)), grade=dict(Counter(x["grade"] for x in rows)),
        heldout_test_issues_excluded=sorted(test_issues), test_issue_lines=0,
        pages_dir="ocr_data/dsk_src_v1/train_pages", ndl_dir="ocr_data/dsk_src_v1/train_ndl",
        manifest_sha256=hashlib.sha256((D / "manifest.jsonl").read_bytes()).hexdigest(),
        lines_jsonl_sha256=hashlib.sha256((X / "lines.jsonl").read_bytes()).hexdigest(),
        test_lines="sg318_v1/test_lines 재사용(시험 60면, 라벨 없음)",
    )
    (X / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in st.items() if k not in ("recorr",)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
