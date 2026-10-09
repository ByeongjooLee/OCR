# -*- coding: utf-8 -*-
"""gaebyeok_v2 학습 LMDB에서 고른 간격으로 뽑은 부분집합(혼합 비율 조절용). 시험 호는 원래 없음.

  작업도구/ocr_train/venv/Scripts/python.exe -X utf8 작업도구/스크립트/gb_subset.py
출력: ocr_data/gaebyeok_v2/train_sub/gaebyeok_half(1/2), gaebyeok_rglyph_fifth(1/5)
"""
import json
import sys
from pathlib import Path

import lmdb

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "ocr_data/gaebyeok_v2"


def sub(src, dst, stride):
    s = lmdb.open(str(V / "train" / src), readonly=True, lock=False)
    dst = V / "train_sub" / dst; dst.mkdir(parents=True, exist_ok=True)
    d = lmdb.open(str(dst), map_size=16 * 1024 ** 3)
    m = 0
    with s.begin() as rt:
        n = int(rt.get(b"num-samples"))
        wt = d.begin(write=True)
        for k in range(1, n + 1, stride):
            m += 1
            wt.put(f"image-{m:09d}".encode(), rt.get(f"image-{k:09d}".encode()))
            wt.put(f"label-{m:09d}".encode(), rt.get(f"label-{k:09d}".encode()))
            if m % 5000 == 0:
                wt.commit(); wt = d.begin(write=True)
        wt.put(b"num-samples", str(m).encode()); wt.commit()
    d.close(); s.close()
    return n, m


if __name__ == "__main__":
    r = {"gaebyeok_half": sub("gaebyeok", "gaebyeok_half", 2), "gaebyeok_rglyph_fifth": sub("gaebyeok_rglyph", "gaebyeok_rglyph_fifth", 5)}
    (V / "train_sub" / "stats.json").write_text(json.dumps(r), encoding="utf-8")
    print(r)
