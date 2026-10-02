# -*- coding: utf-8 -*-
"""검증 분할(사상계 검증 44면, Fable 라벨)에서 CTC+LM 가중치(alpha, beta)를 고른다. 시험세트는 쓰지 않는다.
  python -X utf8 작업도구/스크립트/tune_lm.py <CRNN체크포인트> [줄수]"""
import io, json, sys, itertools
from pathlib import Path
import lmdb, torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "작업도구/ocr_train/parseq")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
import char_lm

def ed(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p = d[:]; d[0] = i
        for j, cb in enumerate(b, 1): d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (ca != cb))
    return d[-1]

GRID_A = [0.2, 0.3, 0.4]
GRID_B = [2.0, 3.0, 4.0]

def main():
    ck, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 600
    m = load_from_checkpoint(ck).eval().cuda(); tf = SceneTextDataModule.get_transform(m.hparams.img_size)
    env = lmdb.open(str(ROOT / "ocr_data/sg318_v1/val/sg318"), readonly=True, lock=False)
    with env.begin() as t:
        N = int(t.get(b"num-samples")); step = max(1, N // n)
        items = [(Image.open(io.BytesIO(t.get(f"image-{k:09d}".encode()))).convert("RGB"), t.get(f"label-{k:09d}".encode()).decode()) for k in range(1, N + 1, step)]
    lps = []
    with torch.inference_mode():
        for k in range(0, len(items), 64):
            x = torch.stack([tf(im) for im, _ in items[k:k + 64]]).cuda()
            lps += list(m(x).log_softmax(-1).float().cpu().numpy())
    itos = m.tokenizer._itos
    lm = char_lm.load()
    res = {}
    greedy = [m.tokenizer.decode(torch.from_numpy(lp[None]).exp())[0][0] for lp in lps]
    tot = sum(len(l) for _, l in items)
    res["greedy"] = sum(ed(g, l) for g, (_, l) in zip(greedy, items)) / tot
    print("greedy", round(res["greedy"] * 100, 2), flush=True)
    for a, b in itertools.product(GRID_A, GRID_B):
        hyp = [char_lm.ctc_beam(lp, itos, lm if a > 0 else None, alpha=a, beta=b) for lp in lps]
        res[f"a{a}_b{b}"] = sum(ed(h, l) for h, (_, l) in zip(hyp, items)) / tot
        print(a, b, round(res[f"a{a}_b{b}"] * 100, 2), flush=True)
    out = ROOT / f"ocr_data/lm/tune_val_{Path(ck).parent.parent.name}_{GRID_A[0]}.json"
    out.write_text(json.dumps(dict(ckpt=ck, lines=len(items), cer=res), indent=1), encoding="utf-8")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8"); main()
