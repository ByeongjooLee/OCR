# -*- coding: utf-8 -*-
"""OCR에서 온 라벨(개벽·코메트)의 점·줄표 반복 폭주 줄을 빼고 LMDB를 복사한다.

  작업도구/ocr_train/venv/Scripts/python.exe -X utf8 작업도구/스크립트/dot_filter.py <원본 LMDB> <새 LMDB>

배경(10-08): 개벽·코메트 라벨의 문장부호는 대조하지 않은 OCR 출력이다. 그래서 목차 점선 같은 줄에
R4의 「······」 폭주가 그대로 들어갔다(라벨 100자 = 최대 길이까지 점). E17이 1930년대에서 나빠진 주원인이다.
뺄 줄: 같은 부호 8자 이상 연속이 있고 (라벨 글자 수 ÷ 줄 길이(글자 칸 수 = 폭/높이))가 2.0을 넘는 줄, 또는 라벨이 100자 이상인 줄.
기준: 개벽 보통 줄의 비는 중앙 1.07, 상위 1% 1.39다. 사람 라벨 k1930의 점 줄은 최대 약 3.3이다(점이 작아 칸당 2~3개).
"""
import io
import json
import re
import sys

import lmdb
from PIL import Image

RUN = re.compile(r"([·…‥.\-―—─])\1{7,}")


def bad(lab, img):
    if len(lab) >= 100:
        return True
    if not RUN.search(lab):
        return False
    w, h = Image.open(io.BytesIO(img)).size
    return len(lab) / max(1e-6, w / h) > 2.0


def main(src, dst):
    s = lmdb.open(src, readonly=True, lock=False)
    d = lmdb.open(dst, map_size=16 * 1024 ** 3)
    m = drop = 0
    ex = []
    with s.begin() as rt:
        n = int(rt.get(b"num-samples"))
        wt = d.begin(write=True)
        for k in range(1, n + 1):
            img, lab = rt.get(f"image-{k:09d}".encode()), rt.get(f"label-{k:09d}".encode())
            if bad(lab.decode(), img):
                drop += 1
                if len(ex) < 5:
                    ex.append(lab.decode()[:60])
                continue
            m += 1
            wt.put(f"image-{m:09d}".encode(), img); wt.put(f"label-{m:09d}".encode(), lab)
            if m % 5000 == 0:
                wt.commit(); wt = d.begin(write=True)
        wt.put(b"num-samples", str(m).encode()); wt.commit()
    d.close(); s.close()
    r = dict(src=src, n=n, kept=m, dropped=drop, examples=ex)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
