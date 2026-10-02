# -*- coding: utf-8 -*-
"""잡지별 판형(레이아웃) 통계와 템플릿 후보 추출.

  python -X utf8 작업도구/스크립트/layout_templates.py

입력
- 사상계: 학습데이터_사상계318/ndl/*.json (NDLOCR-Lite 줄 상자, 시험 호 제외 317면)
- 1930년대 조광·문장·가톨릭청년: 학습데이터_통합/dataset.sqlite의 sources.xml_path(ABBYY FineReader XML, 납품 OCR)
면마다 계산: 본문 영역(정규화), 단(tier) 수와 경계, 단별 열 수, 열 간격(px·본문 폭 대비), 글자 크기(세로줄 폭),
가로줄 비율, 쓰기 방향 분포. 단 = 세로로 서로 겹치는 세로줄의 연결 성분(높이가 본문의 75% 넘는 줄은 단 판정에서 제외).
출력: ocr_data/layout/page_features.jsonl, layout_summary.json, 판형통계.md
"""
import json
import re
import sqlite3
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ocr_data/layout"


def features(lines, W, H):
    """lines: [(x0,y0,x1,y1)]. 반환 dict 또는 None."""
    if len(lines) < 5:
        return None
    vert = [b for b in lines if (b[3] - b[1]) > 1.5 * (b[2] - b[0])]
    hor = [b for b in lines if (b[2] - b[0]) > 1.5 * (b[3] - b[1])]
    xs0 = min(b[0] for b in lines); ys0 = min(b[1] for b in lines)
    xs1 = max(b[2] for b in lines); ys1 = max(b[3] for b in lines)
    bw, bh = max(1, xs1 - xs0), max(1, ys1 - ys0)
    f = dict(W=W, H=H, n_lines=len(lines), n_vert=len(vert), n_hor=len(hor),
             vert_ratio=round(len(vert) / len(lines), 3),
             body=[round(xs0 / W, 3), round(ys0 / H, 3), round(xs1 / W, 3), round(ys1 / H, 3)])
    if len(vert) < 3:
        f["tiers"] = 0
        return f
    tall = [b for b in vert if (b[3] - b[1]) <= 0.75 * bh] or vert
    # 단: 세로 구간이 겹치는 줄끼리 연결
    tall.sort(key=lambda b: b[1])
    parent = list(range(len(tall)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(tall)):
        for j in range(i + 1, len(tall)):
            a, b = tall[i], tall[j]
            ov = min(a[3], b[3]) - max(a[1], b[1])
            if ov > 0.5 * min(a[3] - a[1], b[3] - b[1]):
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, b in enumerate(tall):
        groups[find(i)].append(b)
    tiers = [g for g in groups.values() if len(g) >= 3]
    tiers.sort(key=lambda g: st.median(b[1] for b in g))
    f["tiers"] = len(tiers)
    f["tier_bands"] = [[round(st.median(b[1] for b in g) / H, 3), round(st.median(b[3] for b in g) / H, 3)] for g in tiers]
    f["tier_cols"] = [len(g) for g in tiers]
    pitches, widths = [], []
    for g in tiers:
        cx = sorted((b[0] + b[2]) / 2 for b in g)
        pitches += [b - a for a, b in zip(cx, cx[1:]) if b - a > 0]
        widths += [b[2] - b[0] for b in g]
    if pitches:
        f["col_pitch_px"] = round(st.median(pitches), 1)
        f["col_pitch_rel"] = round(st.median(pitches) / W, 4)
        f["char_w_px"] = round(st.median(widths), 1)
        f["char_w_rel"] = round(st.median(widths) / W, 4)
        f["line_len_rel"] = round(st.median((b[3] - b[1]) / H for g in tiers for b in g), 3)
    return f


def sasanggye():
    out = []
    man = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "학습데이터_사상계318/manifest.jsonl", encoding="utf-8")}
    from PIL import Image
    for jp in sorted((ROOT / "학습데이터_사상계318/ndl").glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        boxes = []
        for blk in d["contents"]:
            for b in blk:
                xs = [p[0] for p in b["boundingBox"]]; ys = [p[1] for p in b["boundingBox"]]
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
        W, H = Image.open(ROOT / "학습데이터_사상계318/pages" / f"{jp.stem}.png").size
        f = features(boxes, W, H)
        if f:
            m = man.get(jp.stem, {})
            f.update(magazine="사상계", page=jp.stem, issue=m.get("issue"), year=None, source="NDLOCR-Lite")
            out.append(f)
    return out


def k1930():
    out = []
    c = sqlite3.connect(ROOT / "학습데이터_통합/dataset.sqlite")
    rows = c.execute("select id, xml_path, magazine, issue_id from sources where xml_path is not null").fetchall()
    for sid, xp, mag, iss in rows:
        try:
            t = open(xp, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for k, pg in enumerate(re.split(r"<page ", t)[1:]):
            m = re.match(r'width="(\d+)" height="(\d+)"', pg)
            if not m:
                continue
            W, H = int(m.group(1)), int(m.group(2))
            boxes = [tuple(int(v) for v in x) for x in re.findall(r'<line l="(\d+)" t="(\d+)" r="(\d+)" b="(\d+)"', pg)]
            f = features(boxes, W, H)
            if f:
                f.update(magazine=mag.replace(" (1)", ""), page=f"{sid}_{k}", issue=iss, source="ABBYY XML")
                out.append(f)
    return out


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["magazine"]].append(r)
    summ = {}
    for mag, rs in by.items():
        tc = Counter(r["tiers"] for r in rs)
        s = dict(pages=len(rs), tiers_dist=dict(sorted(tc.items())),
                 vert_ratio_median=round(st.median(r["vert_ratio"] for r in rs), 3))
        for key in ("col_pitch_rel", "char_w_rel", "line_len_rel"):
            v = [r[key] for r in rs if key in r]
            if v:
                s[key + "_median"] = round(st.median(v), 4)
                s[key + "_iqr"] = [round(sorted(v)[len(v) // 4], 4), round(sorted(v)[3 * len(v) // 4], 4)]
        # 템플릿 후보: 단 수별 대표 단 경계(중앙값)
        tmpl = {}
        for n in sorted(tc):
            if n == 0 or tc[n] < max(5, len(rs) * 0.03):
                continue
            g = [r for r in rs if r["tiers"] == n]
            bands = [[round(st.median(r["tier_bands"][i][j] for r in g), 3) for j in (0, 1)] for i in range(n)]
            cols = [round(st.median(r["tier_cols"][i] for r in g), 1) for i in range(n)]
            tmpl[f"{n}단"] = dict(pages=len(g), share=round(len(g) / len(rs), 3), tier_bands=bands, cols_per_tier=cols,
                                 col_pitch_rel=round(st.median(r["col_pitch_rel"] for r in g if "col_pitch_rel" in r), 4))
        s["templates"] = tmpl
        summ[mag] = s
    return summ


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sasanggye() + k1930()
    with open(OUT / "page_features.jsonl", "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = summarize(rows)
    (OUT / "layout_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
