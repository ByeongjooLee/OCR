# -*- coding: utf-8 -*-
"""『개벽』 면 이미지 → 기울기 보정 → NDL 줄 분할 → KMMOCR 인식 (comet_ocr.py와 같은 절차).

  작업도구\\ocr_train\\venv\\Scripts\\python.exe -X utf8 작업도구\\스크립트\\gb_ocr.py 0010 [--device cuda]

입력: ocr_data/gaebyeok_work/{호}/img/*.jpg (gb_fetch.py)
출력: dsk/{면}.png (보정 면, 회색조), ocr/{면}.json (줄 상자·방향·인식·글자 확률)
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "작업도구/스크립트"))
sys.path.insert(0, str(ROOT / "작업도구/ocr_train/parseq"))
sys.path.insert(0, str(ROOT / "작업도구/ndl_split"))
sys.argv, _argv = [sys.argv[0]], sys.argv  # comet_ocr는 import 때 --issue를 읽으므로 비워 두고 가져온다
from comet_ocr import skew, split_long  # noqa: E402
sys.argv = _argv
CKPT = ROOT / "작업도구/ocr_train/runs/E10_parseq_sghg/E10ep1_parseq.ckpt"  # KMMOCR R4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("issue")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from gb_mag import WORK
    W = WORK / a.issue
    import torch
    from strhub.data.module import SceneTextDataModule
    from strhub.models.utils import load_from_checkpoint
    from ndl_split import NDLSplitter
    torch.set_num_threads(8)
    model = load_from_checkpoint(str(CKPT)).eval().to(a.device)
    tf = SceneTextDataModule.get_transform(model.hparams.img_size)
    ndl = NDLSplitter()
    (W / "dsk").mkdir(exist_ok=True)
    (W / "ocr").mkdir(exist_ok=True)
    for f in sorted((W / "img").glob("*.jpg")):
        out = W / "ocr" / (f.stem + ".json")
        if out.exists():
            continue
        g = Image.open(f).convert("L")
        ah, vh = skew(g, 1)
        av, vv = skew(g, 0)
        ang = ah if vh >= vv else av
        if abs(ang) >= 0.15:
            g = g.rotate(ang, resample=Image.BICUBIC, expand=False, fillcolor=255)
        dp = W / "dsk" / (f.stem + ".png")
        g.save(dp)
        r = ndl.split(str(dp))
        lines, pieces, owner = [], [], []
        for i, ln in enumerate(r["lines"]):
            x0, y0, x1, y1 = [int(v) for v in ln["box"]]
            c = g.crop((max(0, x0 - 2), max(0, y0 - 2), min(g.width, x1 + 2), min(g.height, y1 + 2))).convert("RGB")
            if ln["vertical"]:
                c = c.rotate(90, expand=True)
            for pc in split_long(c, max(200, 30 * c.height)):
                pieces.append(tf(pc)); owner.append(i)
            lines.append(dict(box=[x0, y0, x1, y1], dir="v" if ln["vertical"] else "h",
                              kind=ln.get("type_en", ""), block=ln.get("block"), text="", prob=[]))
        with torch.inference_mode():
            for k in range(0, len(owner), 64):
                x = torch.stack(pieces[k:k + 64]).to(a.device)
                txt, pr = model.tokenizer.decode(model(x).softmax(-1))
                for o, s, q in zip(owner[k:k + 64], txt, pr):
                    lines[o]["text"] += s
                    lines[o]["prob"] += [round(float(v), 3) for v in q.tolist()[:len(s)]]
        out.write_text(json.dumps(dict(page=f.stem, deskew_deg=ang, width=g.width, height=g.height, lines=lines),
                                  ensure_ascii=False), encoding="utf-8")
        print(f.stem, len(lines), ang, "".join(l["text"] for l in lines)[:60], flush=True)


if __name__ == "__main__":
    main()
