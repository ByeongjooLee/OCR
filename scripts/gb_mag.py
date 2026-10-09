# -*- coding: utf-8 -*-
"""한국사DB 원문제공 잡지 설정. gb_fetch/gb_ocr/gb_align/gb_chain이 환경변수 GB_MAG(기본 013=개벽)로 고른다.

  GB_MAG=015 python -X utf8 작업도구/스크립트/gb_fetch.py 0010
개벽(013)은 처음 만든 이름(gaebyeok_work, gaebyeok_lines_*)을 그대로 쓴다.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAGS = {  # 코드: (잡지명, 태그)
    "013": ("개벽", "gb"), "015": ("별건곤", "bg"), "014": ("동광", "dg"), "016": ("삼천리", "sc"),
    "018": ("삼천리문학", "scm"), "019": ("만국부인", "mg"), "017": ("대동아", "dda"),
}
MAG = os.environ.get("GB_MAG", "013")
NAME, TAG = MAGS[MAG]
PFX = f"ma_{MAG}"
WORK = ROOT / ("ocr_data/gaebyeok_work" if MAG == "013" else f"ocr_data/ma{MAG}_work")
REVIEW = ROOT / ("학습데이터_개벽" if MAG == "013" else "학습데이터_한국사DB잡지")


def lines_dir(no):
    return ROOT / (f"ocr_data/gaebyeok_lines_{no}_v1" if MAG == "013" else f"ocr_data/ma{MAG}_lines_{no}_v1")
