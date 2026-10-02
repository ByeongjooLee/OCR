# -*- coding: utf-8 -*-
"""학습한 PARSeq 체크포인트로 시험 60면을 줄 단위 인식 → 면 텍스트 → 채점.

  python -X utf8 작업도구/스크립트/sg_ocr_predict.py <체크포인트.ckpt> <라벨> [--export v1]

- 입력: ocr_data/sg318_<export>/test_lines.jsonl (NDL 줄, 읽기 순서)
- 출력: OCR_사상계/벤치마크_Sol_Astra/outputs_model/<라벨>/<면ID>.txt (줄마다 한 줄)
- 채점 1: evaluate_v2 정책(잠정 기준본 60면), NDLOCR·Fable과 함께. 이체자 접기 전/후(A4, L2) 모두 기록.
- 채점 2: 사람 확정 10면(사상개 10개 대조) 한글·한자 정책 채점(rescore_independent.score).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "OCR_사상계/벤치마크_Sol_Astra"
sys.path.insert(0, str(ROOT / "작업도구/ocr_train/parseq"))
from strhub.data.module import SceneTextDataModule  # noqa: E402
from strhub.models.utils import load_from_checkpoint  # noqa: E402


def split_long(im, max_w):
    """긴 줄(눕힌 세로줄)을 글자 사이 빈 열에서 max_w 이하 조각으로 자른다.
    사상계 학습 줄은 99%가 700px 미만이라, 1단 면의 긴 줄(약 1,300px)을 1024px로 줄이면 글자가 뭉개진다."""
    import math
    import numpy as np
    w = im.width
    if w <= max_w:
        return [im]
    n = math.ceil(w / max_w)
    ink = (np.asarray(im.convert("L")) < 128).sum(axis=0)
    cuts = [0]
    for k in range(1, n):
        c = int(w * k / n)
        lo, hi = max(cuts[-1] + 10, c - w // (4 * n)), min(w - 10, c + w // (4 * n))
        win = ink[lo:hi]
        cuts.append(lo + int(np.argmin(win)) if len(win) else c)
    cuts.append(w)
    return [im.crop((a, 0, b, im.height)) for a, b in zip(cuts, cuts[1:]) if b - a > 4]


def predict(ckpt, export, max_w=0, lm=None, alpha=0.3, beta=3.0):
    X = ROOT / "ocr_data" / f"sg318_{export}"
    model = load_from_checkpoint(str(ckpt)).eval().cuda()
    tf = SceneTextDataModule.get_transform(model.hparams.img_size)
    rows = [json.loads(l) for l in open(X / "test_lines.jsonl", encoding="utf-8")]
    out = {}
    B = 64
    with torch.inference_mode():
        for k in range(0, len(rows), B):
            part = rows[k:k + B]
            pieces, owner = [], []
            for r in part:
                im = Image.open(X / r["image"]).convert("RGB")
                for pc in (split_long(im, max_w) if max_w else [im]):
                    pieces.append(tf(pc)); owner.append(r)
            logits = model(torch.stack(pieces).cuda())
            if lm is not None:  # CTC 모델 + 문자 LM 빔서치(가중치는 검증 분할에서 정함)
                import char_lm
                lps = logits.log_softmax(-1).float().cpu().numpy()
                preds = [char_lm.ctc_beam(lp, model.tokenizer._itos, lm, alpha=alpha, beta=beta) for lp in lps]
            else:
                preds, _ = model.tokenizer.decode(logits.softmax(-1))
            joined = {}
            for r, p in zip(owner, preds):
                joined.setdefault(id(r), [r, ""])[1] += p
            for r, p in joined.values():
                out.setdefault(r["page"], []).append((r["order"], p))
    return {pg: "\n".join(t for _, t in sorted(v)) for pg, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("label")
    ap.add_argument("--export", default="v1")
    ap.add_argument("--split-long", type=int, default=0, help="이 너비(px)보다 긴 줄은 글자 사이에서 잘라 읽는다(0=끔)")
    ap.add_argument("--lm", action="store_true", help="CTC 모델에 문자 5-gram LM 빔서치 적용(ocr_data/lm/char5.pkl)")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--score-only", action="store_true", help="인식은 건너뛰고 outputs_model/<라벨>의 기존 출력만 채점")
    a = ap.parse_args()
    od = BENCH / "outputs_model" / a.label
    if not a.score_only:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        lmobj = None
        if a.lm:
            import char_lm
            lmobj = char_lm.load()
        pages = predict(a.ckpt, a.export, a.split_long, lmobj, a.alpha, a.beta)
        od.mkdir(parents=True, exist_ok=True)
        for pg, t in pages.items():
            (od / f"{pg}.txt").write_text(t, encoding="utf-8")
        print("저장", len(pages), "면 →", od)

    # 채점 1: evaluate_v2 (잠정 기준본 60면)
    sys.path.insert(0, str(BENCH))
    import evaluate_v2 as ev
    ev.OUT = BENCH / "evaluation_model" / a.label
    ev.OUT.mkdir(parents=True, exist_ok=True)
    ev.main([f"{a.label}=outputs_model/{a.label}", "NDLOCR=outputs_baseline/NDLOCR", "Fable_T=outputs_v2/Fable_T", "Opus55_S=outputs_v2/Opus55_S"])
    summ = json.loads((ev.OUT / "요약_v2.json").read_text(encoding="utf-8"))["모델"]
    res = {}
    for lab, v in summ.items():
        t = v["전체"]
        res[lab] = dict(A1=t["A1_CER"], A4_한자=t["A4_한자CER"], L2_이체자접기=t.get("L2_CER"), A1_95CI=t.get("A1_95CI"))

    # 채점 2: 사람 확정 10면
    sys.path.insert(0, os.environ["SG_RESCORE_DIR"])  # 사람 확정 10면 채점기(비공개)
    import rescore_independent as R
    R.REF_DIR = Path(os.environ["SG_HUMAN_REF_DIR"])  # 사람 확정 10면 기준본(비공개)
    human = {}
    for lab, d in [(a.label, od), ("NDLOCR", BENCH / "outputs_baseline/NDLOCR"), ("Fable_T", BENCH / "outputs_v2/Fable_T")]:
        rows = []
        for rp in sorted(R.REF_DIR.glob("sg_*.txt")):
            rows.append(R.score(rp.read_text(encoding="utf-8-sig"), (d / rp.name).read_text(encoding="utf-8"))[0])
        for fold in (False, True):
            rr = [R.score(rp.read_text(encoding="utf-8-sig"), (d / rp.name).read_text(encoding="utf-8"), fold=fold)[0] for rp in sorted(R.REF_DIR.glob("sg_*.txt"))]
            g = R.agg(rr)
            human[f"{lab}{'_이체자접기' if fold else ''}"] = dict(CER=round(g["CER"], 4), 한글=round(g["한글_CER"], 4), 한자=round(g["한자_CER"], 4))
    report = dict(label=a.label, ckpt=str(a.ckpt), 잠정60면=res, 사람확정10면=human)
    (ev.OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
