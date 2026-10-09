# -*- coding: utf-8 -*-
"""한국사DB 잡지 1개 호의 학습 줄(train=True) → 공개 저장소 폴더: 줄 이미지(WebP) + lines.jsonl(라벨·OCR 결과·판정).

  GB_MAG=013 python -X utf8 작업도구/스크립트/gb_publish_imgs.py <저장소 data 폴더> 0010

사용자 결정(10-09): git에는 이미지와 인식 결과를 함께, 학습 줄만, 호별로 묶어 올린다(사상계 v1과 같은 형식).
이미지는 줄 이미지(lines_dir/images/*.png)를 회색조 WebP 품질 90으로 바꾼다(PNG의 약 40%). 세로줄은 원래 방향 그대로다.
시험 호(gb_mag.is_test)는 만들지 않는다. 출력: <data>/<이름>/<호>/lines.jsonl, images/<id>.webp
"""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gb_mag import MAG, lines_dir, is_test  # noqa: E402

KEEP = ("id", "magazine", "issue", "date", "article", "title", "image_file", "ref_page", "box", "dir", "kind",
        "label", "ocr", "grade", "status", "ref_text", "diffs")


def main(dst, no):
    if is_test(no):
        print(no, "test issue — skip"); return
    name = "gaebyeok_lines_v1" if MAG == "013" else f"ma{MAG}_lines_v1"
    src = lines_dir(no)
    out = Path(dst) / name / no
    (out / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "lines.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for l in open(src / "lines.jsonl", encoding="utf-8"):
            x = json.loads(l)
            if not x["train"]:
                continue
            Image.open(src / x["image"]).convert("L").save(out / "images" / f"{x['id']}.webp", "WEBP", quality=90, method=6)
            r = {k: x.get(k) for k in KEEP}
            r["image"] = f"images/{x['id']}.webp"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(no, "train lines", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
