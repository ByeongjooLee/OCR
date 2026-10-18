# -*- coding: utf-8 -*-
"""KMMOCR 사상계 기사 판독분(kmmocr_작업) 중 corr.tsv로 지면을 보고 고친 줄만 학습 줄로 내보낸다 (2026-10-18).

  python -X utf8 작업도구/스크립트/sg_kmmocr_corr_export.py [--name v1] [--dry]

- KMMOCR 초벌 그대로인 줄은 넣지 않는다(자기 출력 재학습 금지, sg_vision_dataset.py register-transcript 쓰지 않음).
- 대상: 진행기록.tsv의 done 행 폴더(계속분 _contNNN 포함), 고정 시험세트 20개 호는 통째로 뺀다.
- 고친 줄 = corr.tsv의 줄 전체 수정(key) 또는 구간 수정(key.N)이 있는 줄. 라벨은 sg_article_ocr.corrected()와 같은 규칙으로 만들고 공백을 뺀다.
- 빼는 줄: <DEL>, 표(「 | 」), 제목·광고·캡션 등 본문 아닌 줄, 가로줄, [?]·□, 61자 이상,
  줄 전체 수정의 글자 수가 초벌과 크게 다른 줄(0.6~1.6배 밖: 읽기 순서·끼워 넣기 정리), 최종 판독문(추가_*.txt)에 그대로 없는 줄(나중에 손으로 더 고친 줄).
- 등급: bronze_kmmocr_corr(Claude가 지면 조각을 보고 고친 줄, 사람 확인 전). 이체자는 sg318과 같이 정자로 통일(원 판독 label_raw 보존).
- 분할: sg318과 같은 호 단위 검증 호(0002·0034·0054·0071·0072·0075)는 val.
출력: ocr_data/sg_kmmocr_corr_<name>/{train,val}/data.mdb, lines.jsonl, charset.txt, stats.json
진행 중인 판독이 끝나면 새 이름(v2…)으로 다시 돌린다(기존 버전 덮어쓰기 금지).
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sg318_export import norm_label, png, write_lmdb  # noqa: E402
from sg_article_ocr import PAGES, WORK, ROOT, corrected, read_corr  # noqa: E402  (함수만 가져다 씀, 고치지 않음)

BENCH = ROOT / "OCR_사상계/벤치마크_Sol_Astra"
VAL_ISSUES = {"0002", "0034", "0054", "0071", "0072", "0075"}   # 학습데이터_사상계318 manifest_summary와 같음


def crop(img, box, d):
    x0, y0, x1, y1 = box
    c = img.crop((max(0, x0 - 2), max(0, y0 - 2), min(img.width, x1 + 2), min(img.height, y1 + 2)))
    return c.rotate(90, expand=True) if d == "v" else c


def squash(t):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", t))


def done_folders():
    """진행기록.tsv done 행 → [(폴더, 최종 판독문 파일, 호)]."""
    rows = [l.rstrip("\n").split("\t") for l in open(WORK / "진행기록.tsv", encoding="utf-8")]
    h = rows[0]
    out = []
    for r in rows[1:]:
        x = dict(zip(h, r))
        if x.get("처리") != "done" or not x.get("파일"):
            continue
        iss = x["호"]
        first = int(re.match(r"\d+", x["면"]).group(0))
        for d in sorted(WORK.glob(f"{iss}_p{first:04d}*")):
            if d.is_dir() and (d / "pages.json").exists() and re.fullmatch(rf"{iss}_p{first:04d}(-\d+)?(_cont\d+)?", d.name):
                out.append((d, ROOT / "OCR_사상계" / x["파일"], iss))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    X = ROOT / "ocr_data" / f"sg_kmmocr_corr_{a.name}"
    if not a.dry and X.exists():
        sys.exit(f"이미 있음(덮어쓰지 않음): {X}")
    test_ids = json.loads((BENCH / "04_고정시험세트.json").read_text(encoding="utf-8"))["frozen_test_ids"]
    test_issues = {t.split("_")[1] for t in test_ids}

    stats, rows, samples, arts = Counter(), [], {"train": [], "val": []}, Counter()
    for d, final, iss in done_folders():
        if iss in test_issues:
            stats["시험호_폴더"] += 1
            continue
        if not (d / "corr.tsv").exists():
            stats["corr없음_폴더"] += 1
            continue
        data = json.loads((d / "pages.json").read_text(encoding="utf-8"))
        corr = read_corr(d / "corr.tsv")
        ftxt = squash(final.read_text(encoding="utf-8")) if final.exists() else ""
        ftxt = ftxt.replace("[?]", "")
        sp = "val" if iss in VAL_ISSUES else "train"
        for p in data["pages"]:
            if p.get("missing"):
                continue
            ip = PAGES / f"sg_{iss}_{p['tag']}.png"
            img = Image.open(ip) if ip.exists() else None
            if img is None or img.size != (p["width"], p["height"]):
                stats["면이미지_불일치"] += sum(1 for l in p["lines"] if f"{p['id']}:{l['n']:03d}" in corr)
                continue
            for l in p["lines"]:
                key = f"{p['id']}:{l['n']:03d}"
                full = key in corr
                span = any(f"{key}.{s}" in corr for s in range(1, len(l.get("spans", [])) + 1))
                if not (full or span):
                    continue
                t = corrected(l, key, corr).lstrip("¶")
                raw = squash(t)
                reason = None
                if t == "<DEL>":
                    reason = "삭제"
                elif " | " in t or "|" in raw:
                    reason = "표"
                elif l["kind"] != "body":
                    reason = "본문아님"
                elif l["dir"] != "v":
                    reason = "가로줄"
                elif "[?]" in t or "□" in t:
                    reason = "불확실글자"
                elif not 1 <= len(raw) <= 60:
                    reason = "길이"
                elif full and not 0.6 <= len(raw) / max(1, len(squash(l["text"]))) <= 1.6:
                    reason = "글자수차이"
                elif raw == squash(l["text"]):
                    reason = "바뀐것없음"
                else:
                    if raw not in ftxt:
                        alt = raw.replace("『", "「").replace("』", "」")
                        if alt in ftxt:
                            raw = alt   # finalize --brackets single(모델의 『』→「」 체계 오류) 적용된 글
                        else:
                            reason = "최종본과다름"
                stats[reason or "사용"] += 1
                if reason:
                    continue
                label = norm_label(raw)
                lid = f"sgk_{iss}_{p['tag']}_L{l['n']:03d}"
                samples[sp].append((png(crop(img, l["box"], "v").convert("RGB")), label))
                arts[final.name] += 1
                rows.append(dict(id=lid, page=f"sg_{iss}_p{p['tag']}", issue=iss, split=sp, box=l["box"], dir=l["dir"],
                                 deskew_deg=p.get("deskew", 0.0), label=label, label_raw=raw, ocr=l["main"], ocr_alt=l["alt"],
                                 minp=l.get("minp"), corr="full" if full else "span", folder=d.name, article=final.stem,
                                 grade="bronze_kmmocr_corr"))
    print(json.dumps(dict(stats), ensure_ascii=False), "train", len(samples["train"]), "val", len(samples["val"]))
    if a.dry:
        return
    if any(r["issue"] in test_issues for r in rows):
        sys.exit("시험 호가 섞였다")
    for s, ss in samples.items():
        write_lmdb(X / s, ss)
    with open(X / "lines.jsonl", "w", encoding="utf-8", newline="\n") as w:
        for x in rows:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    chars = Counter(c for _, lab in samples["train"] for c in lab)
    (X / "charset.txt").write_text("".join(sorted(chars)), encoding="utf-8")
    st = dict(name=f"sg_kmmocr_corr_{a.name}", created="2026-10-18", script="작업도구/스크립트/sg_kmmocr_corr_export.py",
              note="KMMOCR 초벌 중 Claude가 지면 조각을 보고 고친 줄만(초벌 그대로인 줄 제외). 사람 확인 전.",
              filter=dict(stats), train=len(samples["train"]), val=len(samples["val"]),
              train_chars=sum(len(l) for _, l in samples["train"]), charset_size=len(chars),
              articles=len(arts), issues=sorted({r["issue"] for r in rows}), grade=dict(Counter(r["grade"] for r in rows)),
              heldout_test_issues_excluded=sorted(test_issues), test_issue_lines=0,
              lines_jsonl_sha256=hashlib.sha256((X / "lines.jsonl").read_bytes()).hexdigest())
    (X / "stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
