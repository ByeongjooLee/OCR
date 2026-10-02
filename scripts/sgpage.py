# -*- coding: utf-8 -*-
"""sgpage.py ISSUE PAGE [PAGE2] [옵션]   사상계(동방미디어 스캔) 면 렌더

  python sgpage.py 0001 20            인쇄면 20
  python sgpage.py 0001 20 34         20~34면
  python sgpage.py 0001 B001          코드면(0000 표지 / A 속표지 / B 목차 / C·D 광고 / E 판권·편집후기 / X 뒤표지)
  python sgpage.py 0001 20 --bands 0.03,0.48,0.46,0.99    단 경계를 직접 지정(위/아래 단)
  python sgpage.py 0001 20 34 --foot  쪽번호(하단 12%)만 잘라 확인 — 끝면·쪽번호 오식 확인용
  python sgpage.py 0001 20 34 --thumb 저해상 미리보기만(끝면 찾기용, 싸다)

산출: scratchpad/sgpages/<ISSUE>/p0020.png(전면), _bK(단), _bKr/_bKl(단의 좌우 반쪽)
세로쓰기라 **_bKr(오른쪽)부터** 읽는다. 반쪽 컷은 긴 변 1990px에 맞춰 키운다(화면 표시 한도에 맞춤).
원본이 200dpi 흑백이라 2배로 키우고 대비를 고른다. 4배 이상 키워도 정보는 늘지 않는다.
"""
import os, sys, glob
import numpy as np
from PIL import Image, ImageOps, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glyphs import binarize, bands as _bands

SRC = os.environ.get("SG_SCAN_DIR", "")  # 사상계 원본 스캔 폴더(동방미디어, 비공개)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratchpad", "sgpages")
SCALE, FIT = 2.0, 1990

# 파일명 쪽번호가 지면과 뒤바뀐 곳(판독 중 확인) — 인쇄면 → 실제 파일 꼬리
SWAP = {("0014", "0169"): "0170", ("0014", "0170"): "0169"}

def path(iss, page):
    tag = f"{int(page):04d}" if str(page).isdigit() else str(page).upper()
    tag = SWAP.get((iss, tag), tag)
    for ext in ("TIF", "tif", "JPG", "jpg", "GIF", "gif"):
        p = os.path.join(SRC, iss, f"{iss}{tag}.{ext}")
        if os.path.exists(p): return p
    g = glob.glob(os.path.join(SRC, iss, f"{iss}{tag}.*")) + glob.glob(os.path.join(SRC, iss, f"{iss}{tag.lower()}.*"))
    if g: return g[0]
    # 분기별 폴더 등 다른 위치도 찾는다
    for r, _, fs in os.walk(os.path.dirname(os.path.dirname(SRC))):
        for f in fs:
            if os.path.splitext(f)[0].upper() == f"{iss}{tag}".upper(): return os.path.join(r, f)
    raise FileNotFoundError(f"{iss} {tag}: 파일 없음")

def load(iss, page):
    im = Image.open(path(iss, page)).convert("L")
    im = im.resize((int(im.width * SCALE), int(im.height * SCALE)), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=90, threshold=2))
    return ImageOps.autocontrast(im, cutoff=0.5)

def fit(im, long_edge=FIT):
    s = long_edge / max(im.size)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS) if s > 1.02 else im

def seg_of(im, manual=None):
    """단 구간 [(y0,y1)] — 짧은 밴드는 이웃에 합치고, 하나뿐이면 위/아래 2단으로 나눈다"""
    H = im.height
    if manual:
        v = [float(x) for x in manual.split(",")]
        return [(int(v[i] * H), int(v[i + 1] * H)) for i in range(0, len(v) - 1, 2)]
    bs = [list(b) for b in _bands(binarize(im), H)]
    out = []
    for a, b in bs:
        if out and (b - a) < H * .15: out[-1][1] = b        # 짧은 밴드는 앞에 붙임
        elif (b - a) < H * .15 and out: out[-1][1] = b
        else: out.append([a, b])
    out = [(a, b) for a, b in out if b - a > H * .12]
    if len(out) <= 1:                                        # 검출 실패 → 사상계 기본 2단
        out = [(int(H * .03), int(H * .50)), (int(H * .46), int(H * .99))]
    return out

def render(iss, page, manual=None, foot=False, thumb=False):
    im = load(iss, page)
    tag = f"p{int(page):04d}" if str(page).isdigit() else f"x{str(page).upper()}"
    od = os.path.join(OUT, iss); os.makedirs(od, exist_ok=True)
    if thumb:
        t = im.copy(); t.thumbnail((900, 1200), Image.LANCZOS)
        t.save(os.path.join(od, f"{tag}_t.png")); return [f"{tag}_t.png"]
    if foot:
        im.crop((0, int(im.height * .88), im.width, im.height)).save(os.path.join(od, f"{tag}_foot.png"))
        return [f"{tag}_foot.png"]
    outs = [f"{tag}.png"]; im.save(os.path.join(od, f"{tag}.png"))
    segs = seg_of(im, manual)
    for k, (a, b) in enumerate(segs, 1):
        a, b = max(0, a - 15), min(im.height, b + 15)
        tg = f"_b{k}" if len(segs) > 1 else ""
        if tg:
            fit(im.crop((0, a, im.width, b))).save(os.path.join(od, f"{tag}{tg}.png")); outs.append(f"{tag}{tg}.png")
        W = im.width; ov = int(W * .04)
        for side, (x0, x1) in (("r", (W // 2 - ov, W)), ("l", (0, W // 2 + ov))):
            fn = f"{tag}{tg}{side}.png"
            fit(im.crop((x0, a, x1, b))).save(os.path.join(od, fn)); outs.append(fn)
    return outs

if __name__ == "__main__":
    av = sys.argv[1:]
    manual = None
    if "--bands" in av:
        i = av.index("--bands"); manual = av[i + 1]; del av[i:i + 2]
    foot = "--foot" in av; thumb = "--thumb" in av
    av = [x for x in av if not x.startswith("--")]
    iss, a = av[0], av[1]
    rng = range(int(a), int(av[2]) + 1) if len(av) > 2 and a.isdigit() else [a]
    for p in rng:
        try: print(p, render(iss, p, manual, foot, thumb))
        except FileNotFoundError: print(p, "없음")
