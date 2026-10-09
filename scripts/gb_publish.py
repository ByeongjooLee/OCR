# -*- coding: utf-8 -*-
"""한국사DB 잡지 줄 데이터 → 공개 저장소용 메타데이터(호마다 lines/pages .jsonl.gz). 이미지는 넣지 않는다(gb_restore.py로 복원). 시험 호(gb_mag.is_test)는 넣지 않는다.

  GB_MAG=013 python -X utf8 작업도구/스크립트/gb_publish.py <저장소 data 폴더>
출력: <폴더>/{데이터 이름}/{호}.lines.jsonl.gz, {호}.pages.jsonl.gz, index.json
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gb_mag import MAG, NAME, PFX, WORK, lines_dir, ROOT, is_test  # noqa: E402

KEEP = ("id", "magazine", "issue", "date", "article", "title", "image_file", "ref_page", "box", "dir", "kind",
        "image", "label", "ocr", "grade", "status", "train", "ref_text", "diffs")


def main(dst):
    name = "gaebyeok_lines_v1" if MAG == "013" else f"ma{MAG}_lines_v1"
    out = Path(dst) / name
    out.mkdir(parents=True, exist_ok=True)
    idx, tot = [], Counter()
    for d in sorted(ROOT.glob("ocr_data/" + lines_dir("XXXX").name.replace("XXXX", "*"))):
        if not (d / "stats.json").exists():
            continue
        no = d.name.split("_")[-2]
        if is_test(no):   # 시험 호는 논문 게재 전 공개하지 않는다(로컬에만)
            continue
        rows = [json.loads(l) for l in open(d / "lines.jsonl", encoding="utf-8")]
        with gzip.open(out / f"{no}.lines.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as f:
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in KEEP}, ensure_ascii=False) + "\n")
        pages = []
        for p in sorted((WORK / no / "ocr").glob("*.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            pages.append(dict(page=j["page"], deskew_deg=j["deskew_deg"], width=j["width"], height=j["height"],
                              source=f"https://db.history.go.kr/common/imageProxy.do?filePath=/ma/{MAG}/{no}/{j['page']}.jpg"))
        with gzip.open(out / f"{no}.pages.jsonl.gz", "wt", encoding="utf-8", compresslevel=9) as f:
            for p in pages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        g = Counter(r["grade"] for r in rows)
        tot.update(g); tot["pages"] += len(pages)
        idx.append(dict(issue=f"{PFX}_{no}", pages=len(pages), lines=len(rows), grades=dict(g),
                        train_lines=sum(r["train"] for r in rows)))
    (out / "index.json").write_text(json.dumps(dict(magazine=NAME, code=PFX, issues=idx, totals=dict(tot)),
                                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(name, len(idx), dict(tot))


if __name__ == "__main__":
    main(sys.argv[1])
