# -*- coding: utf-8 -*-
"""세로쓰기 면의 기울기 추정과 보정.

방법(일반적인 투영 분산 최대화): 면을 -3°~+3°에서 0.1° 간격으로 돌려 보고, 세로 방향 잉크 투영(열마다 잉크 합)의
분산이 가장 큰 각도를 고른다. 세로줄이 수직이 되면 열 사이 여백이 또렷해져 분산이 커진다.
시험세트로 조정한 값은 없다(탐색 범위·간격은 사전에 정한 일반값).

  python -X utf8 작업도구/스크립트/deskew.py <입력폴더> <출력폴더>   # 보정 이미지와 angles.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def estimate(img, lo=-3.0, hi=3.0, step=0.1):
    g = img.convert("L")
    small = g.resize((g.width // 2, g.height // 2))
    a = np.asarray(small)
    ink = (a < 128).astype(np.float32)
    h, w = ink.shape
    core = Image.fromarray((ink * 255).astype(np.uint8))
    best, best_v = 0.0, -1
    for ang in np.arange(lo, hi + 1e-9, step):
        r = np.asarray(core.rotate(ang, resample=Image.BILINEAR, fillcolor=0), dtype=np.float32)
        r = r[int(h * .05):int(h * .95), int(w * .05):int(w * .95)]
        v = r.sum(axis=0).var()
        if v > best_v:
            best, best_v = float(ang), v
    return round(best, 2)


def rotate(img, ang):
    return img.convert("L").rotate(ang, resample=Image.BICUBIC, expand=False, fillcolor=255)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for p in sorted(src.glob("*.png")):
        im = Image.open(p)
        ang = estimate(im)
        out[p.stem] = ang
        (rotate(im, ang) if abs(ang) >= 0.15 else im.convert("L")).save(dst / p.name)
        print(p.stem, ang, flush=True)
    (dst / "angles.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
