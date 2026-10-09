# -*- coding: utf-8 -*-
"""한국사DB 잡지 줄 이미지 복원: 공개 저장소 메타데이터(lines/pages .jsonl.gz) + 국편 원문 면 이미지 → 줄 이미지.

  GB_MAG=013 python -X utf8 작업도구/스크립트/gb_fetch.py 0010           # 면 이미지 다시 받기(요청 간격 1.5초)
  GB_MAG=013 python -X utf8 작업도구/스크립트/gb_restore.py <메타 폴더> 0010 [--check]

gb_ocr.py와 같은 보정을 한다: 회색조 → |deskew_deg|≥0.15면 BICUBIC 회전(fillcolor 255, expand 없음) → 상자 ±2px 자르기.
--check: 이미 있는 줄 이미지와 픽셀이 같은지 확인만 한다.
"""
import gzip
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gb_mag import WORK, lines_dir  # noqa: E402


def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main(meta, no, check=False):
    meta = Path(meta)
    pages = {p["page"]: p for p in load(meta / f"{no}.pages.jsonl.gz")}
    out = lines_dir(no)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cur, same, diff, n = (None, None), 0, 0, 0
    for x in load(meta / f"{no}.lines.jsonl.gz"):
        pg = x["image_file"][:-4]
        if cur[0] != pg:
            g = Image.open(WORK / no / "img" / x["image_file"]).convert("L")
            a = pages[pg]["deskew_deg"]
            if abs(a) >= 0.15:
                g = g.rotate(a, resample=Image.BICUBIC, expand=False, fillcolor=255)
            cur = (pg, g)
        g = cur[1]
        x0, y0, x1, y1 = x["box"]
        c = g.crop((max(0, x0 - 2), max(0, y0 - 2), min(g.width, x1 + 2), min(g.height, y1 + 2)))
        dst = out / x["image"]
        n += 1
        if check:
            o = Image.open(dst).convert("L")
            if o.size == c.size and ImageChops.difference(o, c).getbbox() is None:
                same += 1
            else:
                diff += 1
        else:
            c.save(dst)
    print(json.dumps(dict(issue=no, lines=n, same=same, diff=diff) if check else dict(issue=no, restored=n)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--check" in sys.argv)
