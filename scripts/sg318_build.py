# -*- coding: utf-8 -*-
"""사상계 학습자료(시험세트와 다른 호 318면) 구축.

  python -X utf8 작업도구/스크립트/sg318_build.py manifest   # 면 목록·전사·원본 경로·분할
  python -X utf8 작업도구/스크립트/sg318_build.py images     # 원본 → PNG 복사(영문 경로 작업폴더 포함)
  python -X utf8 작업도구/스크립트/sg318_build.py ndl        # NDLOCR-Lite 세로줄 검출
  python -X utf8 작업도구/스크립트/sg318_build.py align      # 면 전사 ↔ 세로줄 정렬, 줄 이미지·라벨
  python -X utf8 작업도구/스크립트/sg318_build.py qa         # 육안 점검용 표본 화면

원칙
- 시험세트(04_고정시험세트.json)의 20개 호는 통째로 제외한다(호 단위 분리).
- 학습/검증은 호 단위 고정 해시로 나눈다(약 90/10).
- 원본 TIF와 판독 전사는 수정하지 않는다. 라벨은 기존 생성형 판독 전사(사람 검수 10면 기준 CER 약 3.9%)이므로
  status는 'silver_unreviewed'로 둔다.
"""
import hashlib
import json
import os
import tempfile
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SG = ROOT / "OCR_사상계"
OUT = ROOT / "학습데이터_사상계318"
WORK = Path(os.environ.get("SG_WORK", Path(tempfile.gettempdir()) / "sg318_work"))  # OpenCV 한글 경로 문제 회피용
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sgpage import path as scan_path  # noqa: E402

TEST = json.loads((SG / "벤치마크_Sol_Astra/04_고정시험세트.json").read_text(encoding="utf-8"))["frozen_test_ids"]
TEST_ISSUES = {t.split("_")[1] for t in TEST}
PAGE_RE = re.compile(r"─+\s*p\.\s*(\d+)\s*─+")
HANJA = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def split_of(issue):
    h = int(hashlib.sha256(f"sg318:{issue}".encode()).hexdigest(), 16) % 10
    return "validation" if h == 0 else "train"


def cmd_manifest():
    OUT.mkdir(exist_ok=True)
    rows, seen, dup = [], {}, []
    for f in sorted(SG.glob("*.txt")):
        t = f.read_text(encoding="utf-8")
        m = re.search(r"통권\s*(\d+)호", t[:400])
        if not m:
            continue
        iss = f"{int(m.group(1)):04d}"
        if iss in TEST_ISSUES:
            continue
        nums = PAGE_RE.findall(t)
        bodies = PAGE_RE.split(t)[1::2 + 0] if False else re.split(PAGE_RE, t)
        # re.split with a group returns [head, p1, body1, p2, body2, ...]
        for k in range(1, len(bodies), 2):
            page, body = int(bodies[k]), bodies[k + 1]
            pid = f"sg_{iss}_p{page:04d}"
            body = unicodedata.normalize("NFC", body).strip("\n")
            if pid in seen:
                # 한 면에 두 기사가 걸친 경우: 파일 번호순(=지면순)으로 이어 붙인다
                prev = next(x for x in rows if x["id"] == pid)
                prev["transcript"] += "\n" + body
                prev["transcript_file"] += " + " + f.name
                nows = re.sub(r"\s", "", prev["transcript"])
                prev.update(chars=len(nows), hanja=len(HANJA.findall(nows)),
                            uncertain=prev["transcript"].count("[?]"), illegible=prev["transcript"].count("□"))
                dup.append((pid, seen[pid], f.name))
                continue
            try:
                src = Path(scan_path(iss, page))
            except FileNotFoundError:
                src = None
            seen[pid] = f.name
            nows = re.sub(r"\s", "", body)
            rows.append(dict(id=pid, issue=iss, page=page, split=split_of(iss), transcript_file=f.name,
                             transcript=body, chars=len(nows), hanja=len(HANJA.findall(nows)),
                             uncertain=body.count("[?]"), illegible=body.count("□"),
                             scan=str(src) if src else None,
                             scan_sha256=hashlib.sha256(src.read_bytes()).hexdigest() if src else None,
                             label_status="silver_unreviewed"))
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    iss = sorted({r["issue"] for r in rows})
    summ = dict(pages=len(rows), issues=len(iss), issue_list=iss,
                train_pages=sum(r["split"] == "train" for r in rows),
                validation_pages=sum(r["split"] == "validation" for r in rows),
                validation_issues=sorted({r["issue"] for r in rows if r["split"] == "validation"}),
                chars=sum(r["chars"] for r in rows), hanja=sum(r["hanja"] for r in rows),
                pages_without_scan=[r["id"] for r in rows if not r["scan"]],
                duplicate_pages_skipped=dup, excluded_test_issues=sorted(TEST_ISSUES))
    (OUT / "manifest_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "issue_list"}, ensure_ascii=False))


def load_manifest():
    return [json.loads(l) for l in open(OUT / "manifest.jsonl", encoding="utf-8")]


def cmd_images():
    from PIL import Image
    (OUT / "pages").mkdir(exist_ok=True)
    (WORK / "pages").mkdir(parents=True, exist_ok=True)
    for r in load_manifest():
        if not r["scan"]:
            continue
        dst = OUT / "pages" / f"{r['id']}.png"
        if not dst.exists():
            Image.open(r["scan"]).convert("L").save(dst)
        shutil.copy2(dst, WORK / "pages" / dst.name)
    print("pages:", len(list((OUT / "pages").glob("*.png"))))


def cmd_ndl():
    base = ROOT / "작업도구/ocr_baseline"
    (WORK / "ndl").mkdir(exist_ok=True)
    subprocess.run([str(base / "venv_ndl/Scripts/python.exe"), "ocr.py", "--sourcedir", str(WORK / "pages"),
                    "--output", str(WORK / "ndl"), "--json-only"], cwd=base / "ndlocr-lite/src", check=True)
    (OUT / "ndl").mkdir(exist_ok=True)
    for p in (WORK / "ndl").glob("*.json"):
        shutil.copy2(p, OUT / "ndl" / p.name)
    print("ndl json:", len(list((OUT / "ndl").glob("*.json"))))


# ---------------- 정렬 ----------------

def clean_transcript(t):
    """정렬용: 공백·줄바꿈 제거, [?] 표지 제거(위치는 따로 기록). 원문 전사 자체는 manifest에 그대로 둔다."""
    t = t.replace("[?]", "\x00")
    out, unc = [], set()
    for ch in t:
        if ch == "\x00":
            if out:
                unc.add(len(out) - 1)
            continue
        if ch.isspace():
            continue
        out.append(ch)
    return "".join(out), unc


def align_page(T, lines):
    """T(면 전사)와 NDL 줄들의 이어붙인 문자열 H를 정렬한다.
    한글은 NDL이 엉뚱한 글자로 바꾸므로 '한글↔비한자' 치환 비용을 낮춰 위치 정렬만 하게 하고,
    한자·부호 일치가 실제 닻 역할을 한다. H의 앞뒤(다른 기사 영역)는 무비용으로 건너뛴다."""
    import numpy as np
    H, owner = [], []
    for li, ln in enumerate(lines):
        for ch in ln["text"].replace(" ", ""):
            H.append(ch); owner.append(li)
    n, m = len(T), len(H)
    if n == 0 or m == 0:
        return None
    hj = lambda c: bool(HANJA.match(c))
    INS, DEL = 1.0, 1.0
    D = np.zeros((n + 1, m + 1), dtype=np.float32)
    D[0, :] = 0.0  # H 앞부분 무비용
    D[1:, 0] = np.arange(1, n + 1) * DEL
    Hh = np.array([hj(c) for c in H])
    Harr = np.array(H)
    for i in range(1, n + 1):
        c = T[i - 1]
        if c == Harr[0] and False:
            pass
        eq = (Harr == c)
        if hj(c):
            sub = np.where(eq, 0.0, 1.0)
        else:
            # 한글·부호: 같으면 0, H가 한자가 아니면(가나·잡자) 0.3, H가 한자면 0.8
            sub = np.where(eq, 0.0, np.where(Hh, 0.8, 0.3)).astype(np.float32)
        diag = D[i - 1, :-1] + sub
        up = D[i - 1, 1:] + DEL
        row = np.minimum(diag, up)
        # 삽입(H 글자 건너뛰기)은 좌→우 누적 최소
        cur = np.empty(m + 1, dtype=np.float32)
        cur[0] = D[i, 0]
        for j in range(1, m + 1):
            v = row[j - 1]
            w = cur[j - 1] + INS
            cur[j] = v if v < w else w
        D[i] = cur
    end = int(D[n].argmin())
    # 역추적
    i, j = n, end
    t2h = [None] * n
    while i > 0:
        c = T[i - 1]
        if j > 0:
            if hj(c):
                s = 0.0 if c == H[j - 1] else 1.0
            else:
                s = 0.0 if c == H[j - 1] else (0.8 if Hh[j - 1] else 0.3)
            if abs(D[i, j] - (D[i - 1, j - 1] + s)) < 1e-4:
                t2h[i - 1] = j - 1; i -= 1; j -= 1; continue
        if abs(D[i, j] - (D[i - 1, j] + DEL)) < 1e-4:
            i -= 1; continue
        j -= 1
    return t2h, owner, H, float(D[n, end])


def line_bounds(T, lines):
    """전사 T를 NDL 줄들에 나눠 준다. 반환 (t2h, owner, H, cost, bounds[(줄번호, 시작, 끝)], line_start)."""
    res = align_page(T, lines)
    if not res:
        return None
    t2h, owner, H, cost = res
    # 각 줄에 배정된 전사 구간
    spans = {}
    for ti, hj_ in enumerate(t2h):
        if hj_ is None:
            continue
        li = owner[hj_]
        a, b = spans.get(li, (ti, ti))
        spans[li] = (min(a, ti), max(b, ti))
    # 두 줄 사이에서 대응하지 못한 전사 글자는, 앞 줄 끝과 다음 줄 앞에 남은
    # 미대응 NDL 글자 수의 비율로 나눈다(둘 다 0이면 반씩). 이전 판은 모두 앞 줄에 붙여 줄 끝이 번졌다.
    order = sorted(spans, key=lambda li: spans[li][0])
    line_len = {li: len(lines[li]["text"].replace(" ", "")) for li in order}
    line_start = {}
    pos = 0
    for li, ln in enumerate(lines):
        line_start[li] = pos
        pos += len(ln["text"].replace(" ", ""))
    last_h = {li: max(t2h[k] for k in range(spans[li][0], spans[li][1] + 1) if t2h[k] is not None and owner[t2h[k]] == li) for li in order}
    first_h = {li: min(t2h[k] for k in range(spans[li][0], spans[li][1] + 1) if t2h[k] is not None and owner[t2h[k]] == li) for li in order}
    bounds = []
    a = spans[order[0]][0]
    for k, li in enumerate(order):
        if k + 1 == len(order):
            bounds.append((li, a, spans[li][1]))
            break
        nx = order[k + 1]
        gap_lo, gap_hi = spans[li][1] + 1, spans[nx][0]  # 미대응 전사 글자 [gap_lo, gap_hi)
        gap = max(0, gap_hi - gap_lo)
        tail = line_start[li] + line_len[li] - 1 - last_h[li]  # 앞 줄 끝의 미대응 NDL 글자
        head = first_h[nx] - line_start[nx]                    # 다음 줄 앞의 미대응 NDL 글자
        give = round(gap * tail / (tail + head)) if (tail + head) else gap // 2
        b = gap_lo + give - 1
        bounds.append((li, a, max(a, b)))
        a = b + 1
    return t2h, owner, H, cost, bounds, line_start


def cmd_align():
    from PIL import Image
    (OUT / "lines").mkdir(exist_ok=True)
    out_rows, page_stats = [], []
    for r in load_manifest():
        jp = OUT / "ndl" / f"{r['id']}.json"
        if not jp.exists():
            continue
        lines = [b for blk in json.loads(jp.read_text(encoding="utf-8"))["contents"] for b in blk]
        T, unc = clean_transcript(r["transcript"])
        lb = line_bounds(T, lines)
        if not lb:
            continue
        t2h, owner, H, cost, bounds, line_start = lb
        img = Image.open(OUT / "pages" / f"{r['id']}.png")
        page_hanja = page_hanja_ok = 0
        for li, a, b in bounds:
            ln = lines[li]
            label = T[a:b + 1]
            ntext = ln["text"].replace(" ", "")
            # 품질: 라벨 속 한자 중 같은 줄 NDL 출력에 그대로 있는 비율, 길이 비
            lh = [c for c in label if HANJA.match(c)]
            ok = sum(1 for k2 in range(a, b + 1) if t2h[k2] is not None and owner[t2h[k2]] == li and T[k2] == H[t2h[k2]] and HANJA.match(T[k2]))
            page_hanja += len(lh); page_hanja_ok += ok
            ratio = len(label) / max(1, len(ntext))
            xs = [p[0] for p in ln["boundingBox"]]; ys = [p[1] for p in ln["boundingBox"]]
            box = [min(xs), min(ys), max(xs), max(ys)]
            name = f"{r['id']}_L{li:03d}.png"
            img.crop((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2)).save(OUT / "lines" / name)
            q = "good" if (0.8 <= ratio <= 1.25 and (not lh or ok / len(lh) >= 0.6)) else "check"
            if ln.get("isVertical") != "true":
                q = "check"
            out_rows.append(dict(id=f"{r['id']}_L{li:03d}", page_id=r["id"], issue=r["issue"], split=r["split"],
                                 image=f"lines/{name}", box=box, vertical=ln.get("isVertical") == "true",
                                 tspan=[a, b],  # clean_transcript(공백·[?] 제거) 기준 글자 위치
                                 ndl_start=line_start[li],  # 면 NDL 문자열에서 이 줄의 시작 위치
                                 label=label, ndl_text=ntext, len_ratio=round(ratio, 3),
                                 hanja_in_label=len(lh), hanja_anchor_ok=ok,
                                 uncertain_in_label=sum(1 for k2 in unc if a <= k2 <= b),
                                 quality=q, label_status="silver_unreviewed"))
        covered = sum(b - a + 1 for _, a, b in bounds)
        page_stats.append(dict(id=r["id"], lines_ndl=len(lines), lines_used=len(bounds), chars=len(T),
                               chars_covered=covered, hanja_anchor=round(page_hanja_ok / page_hanja, 3) if page_hanja else None,
                               cost=round(cost, 1)))
        print(r["id"], len(bounds), "lines", page_stats[-1]["hanja_anchor"], flush=True)
    with open(OUT / "lines.jsonl", "w", encoding="utf-8") as w:
        for x in out_rows:
            w.write(json.dumps(x, ensure_ascii=False) + "\n")
    (OUT / "align_pages.json").write_text(json.dumps(page_stats, ensure_ascii=False, indent=1), encoding="utf-8")
    good = [x for x in out_rows if x["quality"] == "good"]
    print(json.dumps(dict(lines=len(out_rows), good=len(good), good_chars=sum(len(x["label"]) for x in good),
                          good_train=sum(x["split"] == "train" for x in good),
                          good_validation=sum(x["split"] == "validation" for x in good)), ensure_ascii=False))


def cmd_qa():
    import base64
    import html as H
    import random
    rows = [json.loads(l) for l in open(OUT / "lines.jsonl", encoding="utf-8")]
    rng = random.Random(20261002)
    sample = rng.sample([x for x in rows if x["quality"] == "good"], 60) + rng.sample([x for x in rows if x["quality"] == "check"], 20)
    cards = []
    for x in sample:
        b = base64.b64encode((OUT / x["image"]).read_bytes()).decode()
        cards.append(f"<div class='c {x['quality']}'><img src='data:image/png;base64,{b}'><div class='t'>"
                     f"<b>{H.escape(x['id'])}</b> · {x['quality']} · 길이비 {x['len_ratio']} · 한자닻 {x['hanja_anchor_ok']}/{x['hanja_in_label']}"
                     f"<p class='lab'>{H.escape(x['label'])}</p><p class='ndl'>NDL: {H.escape(x['ndl_text'])}</p></div></div>")
    page = ("<!doctype html><meta charset='utf-8'><title>줄 정렬 점검</title><style>"
            "body{font:14px 'Malgun Gothic',sans-serif;margin:16px;background:#fafaf7}"
            ".c{display:flex;gap:12px;border-bottom:1px solid #ddd;padding:8px 0}.c img{height:420px;width:auto;border:1px solid #ccc;background:#fff}"
            ".check{background:#fff4f0}.lab{font-size:20px;font-family:Batang,serif;writing-mode:vertical-rl;height:420px;margin:0}"
            ".ndl{color:#888;writing-mode:vertical-rl;height:420px;margin:0}.t{display:flex;gap:10px}.t b{writing-mode:vertical-rl}</style>"
            "<h1>세로줄 이미지 ↔ 전사 정렬 점검 (good 60 · check 20 무작위)</h1>" + "".join(cards))
    (OUT / "정렬점검.html").write_text(page, encoding="utf-8")
    print("저장:", OUT / "정렬점검.html")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"manifest": cmd_manifest, "images": cmd_images, "ndl": cmd_ndl, "align": cmd_align, "qa": cmd_qa}[sys.argv[1]]()
