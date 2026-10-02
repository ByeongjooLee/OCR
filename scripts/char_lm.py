# -*- coding: utf-8 -*-
"""문자 n-gram 언어모델(Witten-Bell 보간)과 CTC 접두사 빔서치. 비생성형, 외부 라이브러리 없이 재현 가능.

  python -X utf8 작업도구/스크립트/char_lm.py build      # 코퍼스 → ocr_data/lm/char5.pkl

코퍼스(공백·줄바꿈 제거, NFC, ○→〇, 이체자 정자 통일 — 학습 라벨과 같은 규칙)
- 자유문학·문예 판독문(OCR_비평, OCR_문예)
- 사상계 판독문(OCR_사상계) 중 **시험세트 20개 호를 통째로 제외**(누출 방지)
- 1930년대 국편 줄 라벨(k1930 학습 분할)
머리말(===, ※, 면 구분선, [유형] 줄)은 뺀다.
"""
import json
import math
import pickle
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sg318_triage import fold  # noqa: E402

LM_PATH = ROOT / "ocr_data/lm/char5.pkl"
ORDER = 5
BOS = "\x02"


def norm(t):
    t = unicodedata.normalize("NFC", t).replace("[?]", "").replace("○", "〇")
    return fold("".join(t.split()))


def article_text(p):
    keep = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith(("=", "※", "─", "—" * 3)) or re.match(r"^\[[^\]]{1,8}\]", s):
            continue
        keep.append(s)
    return norm("".join(keep))


def corpus():
    test = json.loads((ROOT / "OCR_사상계/벤치마크_Sol_Astra/04_고정시험세트.json").read_text(encoding="utf-8"))["frozen_test_ids"]
    test_issues = {t.split("_")[1] for t in test}
    docs, skipped = [], 0
    for d in ("OCR_비평", "OCR_문예"):
        docs += [article_text(p) for p in sorted((ROOT / d).glob("*.txt"))]
    for p in sorted((ROOT / "OCR_사상계").glob("*.txt")):
        head = p.read_text(encoding="utf-8", errors="ignore")[:400]
        m = re.search(r"통권\s*(\d+)호", head) or re.search(r"추가_(\d{4})_", p.name)
        iss = f"{int(m.group(1)):04d}" if m else None
        if iss is None or iss in test_issues:
            skipped += 1
            continue
        docs.append(article_text(p))
    import io
    import lmdb
    env = lmdb.open(str(ROOT / "ocr_data/k1930_v1/train/k1930"), readonly=True, lock=False)
    with env.begin() as t:
        n = int(t.get(b"num-samples"))
        docs.append("".join(t.get(f"label-{k:09d}".encode()).decode() for k in range(1, n + 1)))
    return docs, skipped


class CharLM:
    def __init__(self, counts, order):
        self.c, self.n = counts, order  # c[h] = {ch: count}, h = 길이 0..n-1 문맥
        self.tot = {h: sum(v.values()) for h, v in counts.items()}
        self.typ = {h: len(v) for h, v in counts.items()}
        self.V = len(counts[""]) + 1
        self.cache = {}

    def prob(self, hist, ch):
        hist = hist[-(self.n - 1):]
        key = (hist, ch)
        if key in self.cache:
            return self.cache[key]
        p = 1.0 / self.V
        for k in range(0, len(hist) + 1):
            h = hist[len(hist) - k:] if k else ""
            if h not in self.c:
                break
            T, N = self.typ[h], self.tot[h]
            lam = N / (N + T)
            p = lam * self.c[h].get(ch, 0) / N + (1 - lam) * p
        if len(self.cache) < 2_000_000:
            self.cache[key] = p
        return p

    def logp(self, hist, ch):
        return math.log(self.prob(hist, ch))


def build():
    docs, skipped = corpus()
    counts = defaultdict(lambda: defaultdict(int))
    nchar = 0
    for d in docs:
        s = BOS * (ORDER - 1) + d
        nchar += len(d)
        for i in range(ORDER - 1, len(s)):
            for k in range(0, ORDER):
                counts[s[i - k:i]][s[i]] += 1
    counts = {h: dict(v) for h, v in counts.items()}
    LM_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LM_PATH, "wb") as f:
        pickle.dump({"order": ORDER, "counts": counts}, f, protocol=4)
    meta = dict(docs=len(docs), chars=nchar, contexts=len(counts), skipped_test_issue_files=skipped, order=ORDER)
    (LM_PATH.parent / "char5_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


def load():
    with open(LM_PATH, "rb") as f:
        d = pickle.load(f)
    return CharLM(d["counts"], d["order"])


def ctc_beam(logprobs, itos, lm=None, alpha=0.5, beta=1.0, beam=10, topk=12, blank=0):
    """CTC 접두사 빔서치 + 문자 LM 얕은 결합(shallow fusion).
    logprobs: (T, C) numpy 로그확률. 점수 = log P_ctc + alpha·log P_lm + beta·길이."""
    import numpy as np
    NEG = -1e30

    def lse(a, b):
        if a < b:
            a, b = b, a
        return a if b <= NEG else a + math.log1p(math.exp(b - a))

    beams = {"": (0.0, NEG, 0.0)}  # 접두사 → (p_blank, p_nonblank, lm_score)
    lb_skip = math.log(0.999)
    for t in range(logprobs.shape[0]):
        row = logprobs[t]
        if row[blank] > lb_skip:
            # 빈칸이 거의 확실한 프레임: 모든 빔을 빈칸 상태로 넘긴다(근사, 속도용)
            beams = {k: (lse(pb, pnb) + row[blank], NEG, lms) for k, (pb, pnb, lms) in beams.items()}
            continue
        cand = np.argpartition(-row, topk)[:topk]
        nb = defaultdict(lambda: [NEG, NEG, 0.0])
        for pre, (pb, pnb, lms) in beams.items():
            tot = lse(pb, pnb)
            # 빈칸
            e = nb[pre]; e[0] = lse(e[0], tot + row[blank]); e[2] = lms
            for c in cand:
                if c == blank:
                    continue
                ch = itos[c]
                p = row[c]
                if pre and pre[-1] == ch:
                    # 반복: 빈칸 뒤일 때만 새 글자
                    e2 = nb[pre + ch]
                    if e2[2] == 0.0 and lm is not None:
                        e2[2] = lms + alpha * lm.logp(BOS * 4 + pre, ch) + beta
                    elif lm is None:
                        e2[2] = lms + beta
                    e2[1] = lse(e2[1], pb + p)
                    e = nb[pre]; e[1] = lse(e[1], pnb + p)
                else:
                    e2 = nb[pre + ch]
                    if e2[2] == 0.0:
                        e2[2] = lms + (alpha * lm.logp(BOS * 4 + pre, ch) if lm is not None else 0.0) + beta
                    e2[1] = lse(e2[1], tot + p)
        beams = dict(sorted(((k, tuple(v)) for k, v in nb.items()), key=lambda kv: -(lse(kv[1][0], kv[1][1]) + kv[1][2]))[:beam])
    best = max(beams.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + kv[1][2])
    return best[0]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"build": build}[sys.argv[1]]()
