# -*- coding: utf-8 -*-
"""사상계 318면 줄 라벨 새 버전: Fable T 라벨(sg318_v2dsk와 같은 원천) 위에 KMMOCR 대조 재교정의 지면 확인 판정을 얹는다 (2026-10-18).

  python -X utf8 작업도구/스크립트/sg318_export_recorr.py [--name v3dsk] [--dry]

재교정(sg_audit_fix.py, 1~9차)은 기존 생성형 전사를 고쳤고, 학습 라벨은 Fable T 재판독이 원천이라 그대로였다.
판정마다 지면 조각을 직접 보았으므로 그 자리의 인쇄 글자는 확정이다. 이것을 Fable 라벨에 옮긴다.
- 반영(OCR·FIX·유형 일괄): 지면 = 반영한 글자. Fable이 기존 전사와 같으면 바꾸고, 이미 지면과 같으면 그대로, 둘 다와 다르면 지면 글자로 바꾼다.
- 기존(지면 = 기존 전사): Fable이 OCR 쪽이거나 다른 글자면 기존 전사 글자로 바꾼다.
- 불명: 그 글자를 불확실로 표시해 해당 줄을 학습에서 뺀다(사람 검수 후보).
자리 찾기: 판정 기록의 앞뒤 문맥(ctx_existing, 앞뒤 5자)으로 Fable 문자열에서 한 곳만 맞을 때만 쓴다. 못 찾은 자리는 기록만 남긴다.
나머지는 sg318_export.py와 같다(정렬·품질 필터·이체자 정자 통일·세로줄 90° 회전, 기울기 보정 면 dsk_src_v1).
출력: ocr_data/sg318_<name>/{train,val}/data.mdb, lines.jsonl(label_prev·fixes·grade), overlay_log.jsonl, charset.txt, stats.json
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sg318_build import HANJA, OUT as D, ROOT, clean_transcript, line_bounds, load_manifest  # noqa: E402
from sg318_export import crop, norm_label, png, write_lmdb  # noqa: E402

FIX = ROOT / "OCR_사상계/kmmocr_작업/감사_기존전사/재교정"
KEEP = re.compile(r"[\w가-힣㐀-鿿豈-﫿]")   # sg_audit_existing.KEEP과 같음(가-힣·㐀-鿿·豈-﫿)
BENCH = ROOT / "OCR_사상계/벤치마크_Sol_Astra"


def keep_only(s):
    return "".join(c for c in unicodedata.normalize("NFC", s) if KEEP.match(c))


def load_decisions():
    """1~9차 작업목록(판정) + 적용기록(실제 넣은 글자) → [{round, id, file, page, existing, target, kind, ctx}]."""
    out = []
    for d in [FIX] + sorted(p for p in FIX.iterdir() if p.is_dir() and p.name.endswith("차")):
        rnd = "1차" if d == FIX else d.name
        applied = {}
        for l in open(d / "적용기록.jsonl", encoding="utf-8"):
            x = json.loads(l)
            applied[x["id"]] = x
        with open(d / "작업목록.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if r["id"] in applied:
                    kind, target = "fix", keep_only(applied[r["id"]]["applied"])
                    exf = keep_only(applied[r["id"]]["existing"])
                    if exf != r["existing"]:
                        # 띄어쓰기 낀 자리를 손으로 반영한 기록: 넣은 글자가 앞뒤 글자까지 덮는다 → 공통 앞뒤를 덜어 낸다
                        p = 0
                        while p < min(len(exf), len(target)) and exf[p] == target[p]:
                            p += 1
                        q = 0
                        while q < min(len(exf), len(target)) - p and exf[-1 - q] == target[-1 - q]:
                            q += 1
                        core_ex, core_ap = exf[p:len(exf) - q], target[p:len(target) - q]
                        if core_ex != r["existing"]:
                            continue   # 판정 자리와 손 반영 범위가 맞지 않음 → 쓰지 않음
                        target = core_ap
                elif r["verdict"] == "기존":
                    kind, target = "keep", r["existing"]
                elif r["verdict"] == "불명":
                    kind, target = "unclear", None
                else:
                    continue   # 반영 안 된 auto 예외 등
                out.append(dict(round=rnd, id=r["id"], file=r["file"], page=r["page"], existing=r["existing"],
                                ocr=r["ocr"], target=target, kind=kind, ctx=r["ctx_existing"]))
    return out


def split_ctx(ctx, ex):
    """ctx = ref[i1-5:i2+5] → (왼쪽, 오른쪽). 면 첫머리에서는 왼쪽이 5자보다 짧다."""
    if ctx[5:5 + len(ex)] == ex:
        return ctx[:5], ctx[5 + len(ex):]
    for L in range(min(5, len(ctx)), -1, -1):
        if ctx[L:L + len(ex)] == ex and len(ctx) - L - len(ex) <= 5:
            return ctx[:L], ctx[L + len(ex):]
    return None


def find_all(s, sub):
    i, res = s.find(sub), []
    while i >= 0:
        res.append(i); i = s.find(sub, i + 1)
    return res


def locate(K, dec):
    """K(Fable 면 KEEP 글자열)에서 판정 자리를 찾는다 → (상태, 시작, 끝) — K 위치 [시작, 끝)."""
    sp = split_ctx(dec["ctx"], dec["existing"])
    if not sp:
        return "ctx_bad", None, None
    left, right = sp
    ex = dec["existing"]
    alt = dec["target"] if dec["kind"] == "fix" else dec["ocr"]
    hit = find_all(K, left + ex + right)
    if len(hit) == 1:
        return "fable=existing", hit[0] + len(left), hit[0] + len(left) + len(ex)
    if alt is not None:
        hit = find_all(K, left + alt + right)
        if len(hit) == 1:
            return ("fable=print" if dec["kind"] == "fix" else "fable=ocr"), hit[0] + len(left), hit[0] + len(left) + len(alt)
    # 앞뒤 닻(각 3자 이상)만 맞고 가운데가 다른 경우
    la, ra = left[-3:], right[:3]
    if len(la) < 3 or len(ra) < 3:
        return "notfound", None, None
    cands = []
    for i in find_all(K, la):
        s = i + len(la)
        for gap in range(0, max(len(ex), len(alt or "")) + 3):
            if K[s + gap:s + gap + len(ra)] == ra:
                cands.append((s, s + gap)); break
    if len(cands) == 1:
        return "fable=other", cands[0][0], cands[0][1]
    return "notfound", None, None


def overlay(F, unc, decs):
    """F(Fable 정렬용 문자열), unc([?] 위치) + 판정 → (새 F, 새 unc, fixed 위치, 불명 위치, 기록)."""
    kidx = [i for i, c in enumerate(F) if len(unicodedata.normalize("NFC", c)) == 1 and KEEP.match(unicodedata.normalize("NFC", c))]
    K = "".join(unicodedata.normalize("NFC", F[i]) for i in kidx)
    toks = [dict(c=c, unc=i in unc, fix=False, unclear=False) for i, c in enumerate(F)]
    edits, log = [], []
    for d in decs:
        st, a, b = locate(K, d)
        rec = dict(round=d["round"], id=d["id"], file=d["file"], kind=d["kind"], existing=d["existing"], ocr=d["ocr"],
                   target=d["target"], status=st)
        if a is None:
            log.append(rec); continue
        fable = K[a:b]
        rec["fable"] = fable
        if a < b:
            fa, fb = kidx[a], kidx[b - 1] + 1
            if any(not KEEP.match(unicodedata.normalize("NFC", F[i])) for i in range(fa, fb)):
                rec["status"] = "span_has_punct"; log.append(rec); continue
        else:
            fa = fb = kidx[a] if a < len(kidx) else len(F)
        if d["kind"] == "unclear":
            edits.append((fa, fb, None, "unclear"))
            rec["action"] = "unclear"
        elif fable == d["target"]:
            rec["action"] = "same"
        else:
            edits.append((fa, fb, d["target"], "fix"))
            rec["action"] = "replace"
        log.append(rec)
    # 겹치는 편집은 뒤의 것을 버린다
    edits.sort(key=lambda e: (e[0], e[1]))
    kept, last = [], -1
    for e in edits:
        if e[0] < last or (kept and e[0] == kept[-1][0] and e[0] == e[1] == kept[-1][1]):
            continue
        kept.append(e); last = max(last, e[1])
    for fa, fb, rep, kind in sorted(kept, key=lambda e: -e[0]):
        if kind == "unclear":
            span = range(fa, fb) if fb > fa else range(max(0, fa - 1), min(len(toks), fa + 1))
            for i in span:
                toks[i]["unclear"] = True
        else:
            toks[fa:fb] = [dict(c=ch, unc=False, fix=True, unclear=False) for ch in rep]
            if not rep and fa > 0:   # 삭제: 앞 글자에 표시
                toks[fa - 1]["fix"] = True
    newF = "".join(t["c"] for t in toks)
    return (newF, {i for i, t in enumerate(toks) if t["unc"]}, {i for i, t in enumerate(toks) if t["fix"]},
            {i for i, t in enumerate(toks) if t["unclear"]}, log)


def export_page(F, unc, lines, extra=None):
    """sg318_export.py의 줄 나누기·필터 그대로. extra: (fixed, unclear) 위치 집합."""
    lb = line_bounds(F, lines)
    if not lb:
        return []
    t2h, owner, H, _, bounds, _ = lb
    out = []
    for li, s0, s1 in bounds:
        ln = lines[li]
        raw = F[s0:s1 + 1]
        label = norm_label(raw)
        ntext = ln["text"].replace(" ", "")
        vertical = ln.get("isVertical") == "true"
        lh = [k for k in range(s0, s1 + 1) if HANJA.match(F[k])]
        ok = sum(1 for k in lh if t2h[k] is not None and owner[t2h[k]] == li and F[k] == H[t2h[k]])
        ratio = len(raw) / max(1, len(ntext))
        fixes = unclear = 0
        if extra:
            fixes = sum(1 for k in range(s0, s1 + 1) if k in extra[0])
            unclear = sum(1 for k in range(s0, s1 + 1) if k in extra[1])
        reason = None
        if not vertical:
            reason = "가로줄"
        elif not (0.8 <= ratio <= 1.25) or (lh and ok / len(lh) < 0.6):
            reason = "정렬품질"
        elif "□" in raw or any(s0 <= k <= s1 for k in unc):
            reason = "불확실글자"
        elif unclear:
            reason = "재교정불명"
        elif not 1 <= len(label) <= 60:
            reason = "길이"
        xs = [p[0] for p in ln["boundingBox"]]; ys = [p[1] for p in ln["boundingBox"]]
        out.append(dict(li=li, label=label, reason=reason, fixes=fixes, box=[min(xs), min(ys), max(xs), max(ys)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v3dsk")
    ap.add_argument("--pages", default=str(ROOT / "ocr_data/dsk_src_v1/train_pages"))
    ap.add_argument("--ndl", default=str(ROOT / "ocr_data/dsk_src_v1/train_ndl"))
    ap.add_argument("--dry", action="store_true", help="겹치기 통계만")
    a = ap.parse_args()
    PAGES, NDLD = Path(a.pages), Path(a.ndl)
    X = ROOT / "ocr_data" / f"sg318_{a.name}"
    if not a.dry and X.exists():
        sys.exit(f"이미 있음(덮어쓰지 않음): {X}")
    test_ids = json.loads((BENCH / "04_고정시험세트.json").read_text(encoding="utf-8"))["frozen_test_ids"]
    test_issues = {t.split("_")[1] for t in test_ids}

    decs = load_decisions()
    by_key = {}
    for d in decs:
        by_key.setdefault((d["file"], d["page"]), []).append(d)
    split_samples = {"train": [], "val": []}
    stats, ostat, rows, logs = Counter(), Counter(), [], []
    for r in load_manifest():
        if r["issue"] in test_issues:
            sys.exit(f"시험 호가 manifest에 있다: {r['id']}")
        fp, jp = D / "fable_T" / f"{r['id']}.txt", NDLD / f"{r['id']}.json"
        if not fp.exists() or not jp.exists():
            continue
        F0, unc0 = clean_transcript(fp.read_text(encoding="utf-8"))
        files = r["transcript_file"].split(" + ")
        pdecs = [d for f in files for d in by_key.get((f, f"{r['page']:04d}"), [])]
        F, unc, fixed, unclear, log = overlay(F0, unc0, pdecs)
        for x in log:
            x["page_id"] = r["id"]
            ostat[(x["kind"], x["status"], x.get("action", "-"))] += 1
        logs += log
        lines = [b for blk in json.loads(jp.read_text(encoding="utf-8"))["contents"] for b in blk]
        new = export_page(F, unc, lines, (fixed, unclear))
        prev = {x["li"]: x for x in export_page(F0, unc0, lines)} if pdecs else None
        if a.dry:
            for x in new:
                stats[x["reason"] or "사용"] += 1
                if x["fixes"] and not x["reason"]:
                    stats["사용_고친줄"] += 1
            continue
        img = Image.open(PAGES / f"{r['id']}.png")
        sp = "val" if r["split"] == "validation" else "train"
        for x in new:
            stats[x["reason"] or "사용"] += 1
            if x["reason"]:
                continue
            p = prev.get(x["li"]) if prev is not None else x
            label_prev = p["label"] if p and not p["reason"] else None
            changed = label_prev != x["label"]
            stats["사용_라벨바뀜" if changed else "사용_라벨같음"] += 1
            split_samples[sp].append((png(crop(img, x["box"], True)), x["label"]))
            rows.append(dict(id=f"{r['id']}_L{x['li']:03d}", page=r["id"], issue=r["issue"], split=sp, box=x["box"],
                             label=x["label"], label_prev=label_prev, recorr_fixes=x["fixes"],
                             grade="silver_fixed" if x["fixes"] else "silver"))
    print("판정 겹치기:", len(decs), "건 중 318면 해당", len(logs))
    for k, v in sorted(ostat.items(), key=lambda kv: -kv[1]):
        print("  ", k, v)
    print("줄:", dict(stats))
    if a.dry:
        return
    for sp, ss in split_samples.items():
        write_lmdb(X / sp, ss)
    with open(X / "lines.jsonl", "w", encoding="utf-8", newline="\n") as w:
        for x in rows:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(X / "overlay_log.jsonl", "w", encoding="utf-8", newline="\n") as w:
        for x in logs:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    chars = Counter(c for _, lab in split_samples["train"] for c in lab)
    charset = "".join(sorted(chars))
    (X / "charset.txt").write_text(charset, encoding="utf-8")
    st = dict(
        name=f"sg318_{a.name}", created="2026-10-18", script="작업도구/스크립트/sg318_export_recorr.py",
        label_source="fable_T + KMMOCR 대조 재교정 지면 판정(1~9차)", base_version="sg318_v2dsk",
        filter={k: v for k, v in stats.items() if not k.startswith("사용_")} | {"사용": stats["사용"]},
        label_changed_vs_prev=stats["사용_라벨바뀜"], label_same_vs_prev=stats["사용_라벨같음"],
        train=len(split_samples["train"]), val=len(split_samples["val"]),
        train_chars=sum(len(l) for _, l in split_samples["train"]), charset_size=len(charset),
        hanja_types=sum(1 for c in charset if HANJA.match(c)),
        grade=dict(Counter(x["grade"] for x in rows)),
        overlay={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in ostat.items()},
        heldout_test_issues_excluded=sorted(test_issues), test_issue_lines=0,
        pages_dir="ocr_data/dsk_src_v1/train_pages", ndl_dir="ocr_data/dsk_src_v1/train_ndl",
        manifest_sha256=hashlib.sha256((D / "manifest.jsonl").read_bytes()).hexdigest(),
        lines_jsonl_sha256=hashlib.sha256((X / "lines.jsonl").read_bytes()).hexdigest(),
        test_lines="sg318_v1/test_lines 재사용(시험 60면, 라벨 없음)",
    )
    (X / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in st.items() if k != "overlay"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
