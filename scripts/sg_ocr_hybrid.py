# -*- coding: utf-8 -*-
"""결합 실험: 학습 모델(한글에 강함) + NDLOCR-Lite(한자에 강함), 같은 NDL 줄 단위로 합친다.

  python -X utf8 작업도구/스크립트/sg_ocr_hybrid.py <모델라벨> <출력라벨>

규칙(줄마다): 모델 출력과 NDL 출력을 글자 정렬해서, 치환 자리 가운데 **두 쪽 모두 한자**인 곳만 NDL 글자로 바꾼다.
NDL은 한글을 엉뚱한 한자·가나로 읽으므로, 모델이 한글로 읽은 자리는 건드리지 않는다. 결과는 이체자 정자 통일.
생성형 모델은 쓰지 않는다.
"""
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sg318_build import HANJA, ROOT  # noqa: E402
from sg318_triage import fold  # noqa: E402

BENCH = ROOT / "OCR_사상계/벤치마크_Sol_Astra"
NDL_DIR = sys.argv[3] if len(sys.argv) > 3 else "outputs_baseline/NDLOCR"  # 기울기 보정판은 outputs_baseline/NDLOCR_deskew


def merge(m, n):
    out, changed = [], 0
    sm = difflib.SequenceMatcher(None, m, n, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace" and (i2 - i1) == (j2 - j1):
            for a, b in zip(m[i1:i2], n[j1:j2]):
                if HANJA.match(a) and HANJA.match(b):
                    out.append(b); changed += a != b
                else:
                    out.append(a)
        else:
            out.append(m[i1:i2])
    return fold("".join(out)), changed


def main():
    src, dst = sys.argv[1], sys.argv[2]
    od = BENCH / "outputs_model" / dst
    od.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in sorted((BENCH / "outputs_model" / src).glob("sg_*.txt")):
        mlines = f.read_text(encoding="utf-8").split("\n")
        nd = json.loads((BENCH / NDL_DIR / f.name.replace(".txt", ".json")).read_text(encoding="utf-8"))
        nlines = [b["text"].replace(" ", "") for blk in nd["contents"] for b in blk]
        assert len(mlines) == len(nlines), (f.name, len(mlines), len(nlines))
        res = []
        for m, n in zip(mlines, nlines):
            t, c = merge(m, n); res.append(t); total += c
        (od / f.name).write_text("\n".join(res), encoding="utf-8")
    print("NDL 한자로 바꾼 글자", total)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
