# -*- coding: utf-8 -*-
"""『개벽』 호별 줄 데이터(gaebyeok_lines_{호}_v1) → 학습·시험 LMDB ocr_data/gaebyeok_vN.

  작업도구/ocr_train/venv/Scripts/python.exe -X utf8 작업도구/스크립트/gb_lmdb.py v1

형식·규칙은 comet_lmdb_v3.py와 같다(train=True 줄만, 세로줄은 반시계 90°, 정답 k1930_eval.norm, rglyph 가로 재배열 추가).
시험 격리: 호 통째로 뗀다(같은 기사가 학습·시험에 나뉘지 않게). TEST 호는 버전이 바뀌어도 고정한다.
stats.json이 있는 호(내보내기 끝난 호)만 넣고, 넣은 호 목록을 stats에 적는다. 이미 있는 버전은 덮어쓰지 않는다.
"""
import hashlib
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmdb  # noqa: E402
from comet_lmdb import norm, RG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEST = {"0050", "0150", "0250", "0350", "0450", "0550", "0650", "0740"}   # 5·15·25·35·45·55·65호 + 신간 2호


class Sink:
    """LMDB에 흘려 쓰기(전부 메모리에 들지 않게). map_size는 넉넉히 잡고, 끝나면 실제 크기만 남는다."""
    def __init__(self, dest, gb):
        dest.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(str(dest), map_size=gb * 1024 ** 3)
        self.t, self.n, self.chars = self.env.begin(write=True), 0, 0

    def put(self, img, lab):
        self.n += 1; self.chars += len(lab)
        self.t.put(f"image-{self.n:09d}".encode(), img); self.t.put(f"label-{self.n:09d}".encode(), lab.encode("utf-8"))
        if self.n % 5000 == 0:
            self.t.commit(); self.t = self.env.begin(write=True)

    def close(self):
        self.t.put(b"num-samples", str(self.n).encode()); self.t.commit(); self.env.sync(); self.env.close()


def main(ver):
    out = ROOT / f"ocr_data/gaebyeok_{ver}"
    if out.exists():
        sys.exit(f"{out} 이미 있음 — 새 버전 이름을 쓴다")
    rng = random.Random(20261007)
    issues = sorted(p.name.split("_")[2] for p in (ROOT / "ocr_data").glob("gaebyeok_lines_*_v1") if (p / "stats.json").exists())
    stats = {"version": out.name, "issues": issues, "test_issues": sorted(TEST & set(issues)),
             "test_issues_fixed": sorted(TEST), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "rules": "학습데이터_개벽/README.md", "parts": {}}
    for split in ("train", "test"):
        gb = 20 if split == "train" else 4
        plain, rg, hanja = Sink(out / split / "gaebyeok", gb), Sink(out / split / "gaebyeok_rglyph", gb), 0
        for no in issues:
            if (no in TEST) != (split == "test"):
                continue
            src = ROOT / f"ocr_data/gaebyeok_lines_{no}_v1"
            for x in map(json.loads, open(src / "lines.jsonl", encoding="utf-8")):
                if not x["train"] or x["dir"] not in ("v", "h"):
                    continue
                lab = norm(x["label"])
                if not lab:
                    continue
                im = Image.open(src / x["image"]).convert("RGB")
                if x["dir"] == "v":
                    im = im.rotate(90, expand=True)
                buf = io.BytesIO(); im.save(buf, "PNG"); b = buf.getvalue()
                plain.put(b, lab); hanja += any("一" <= c <= "鿿" for c in lab)
                if x["dir"] == "v":
                    try:
                        r = RG.relayout(b, lab, rng)
                    except ValueError:
                        r = None
                    if r:
                        rg.put(r, lab)
            print(split, no, plain.n, rg.n, flush=True)
        stats["parts"][split] = dict(lines=plain.n, chars=plain.chars, rglyph=rg.n, hanja_lines=hanja)
        plain.close(); rg.close()
    json.dump(stats, open(out / "stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
