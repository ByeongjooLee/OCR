# -*- coding: utf-8 -*-
"""『개벽』 OCR 줄 ↔ 한국사DB 입력문 대조 → 줄 라벨·등급, 학습데이터 내보내기, 보류 줄 검수 화면.

  python -X utf8 작업도구/스크립트/gb_align.py 0010 [--export]

코메트(comet_align.py)와 달리 입력문이 원표기(옛 맞춤법·한자) 그대로다. 그래서 글자끼리 바로 맞춘다.
입력문이 원본과 다른 점(표기 차이, 오답 아님): 현대식 띄어쓰기, 。 생략, 'ㅣ'→'-', 문장부호 일부.
  → 대조는 한글·한자·영숫자만 쓰고, 라벨의 문장부호는 OCR 것을 둔다(확신 낮으면 보류).
판정
  ref_match : 글자가 입력문과 모두 같음 → 라벨 = OCR
  ref_fixed : 입력문으로 고친 글자가 2자 이하이고 줄의 10% 이하 → 라벨 = 고친 OCR
  review    : 그 밖(차이 많음·OCR에만 있는 확신 높은 글자·누락 여럿·이체자 확신 낮음·부호 확신 낮음)
  no_ref    : 입력문에서 찾지 못한 줄(광고·목차·판권·입력 생략분)
  short     : 대조 글자 4자 미만이고 앞줄 위치로도 못 맞춘 줄
출력: ocr_data/gaebyeok_work/{호}/align.jsonl, align_stats.json
--export: ocr_data/gaebyeok_lines_{호}_v1/(images/, lines.jsonl, stats.json), 학습데이터_개벽/검수_gb{호}_보류.html
"""
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NO = sys.argv[1]
from gb_mag import WORK, NAME, TAG, PFX, REVIEW, lines_dir  # noqa: E402
W = WORK / NO
sys.path.insert(0, str(Path(__file__).resolve().parent))
_argv, sys.argv = sys.argv, [sys.argv[0]]
from comet_align import readings, norm  # noqa: E402  (한자 독음, 두음 무시)
sys.argv = _argv

HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
HANG = re.compile(r"[가-힣]")
def is_han(c): return bool(HAN.match(c))
def is_hang(c): return bool(HANG.match(c))
def is_key(c): return is_han(c) or is_hang(c) or (c.isascii() and c.isalnum())
NUM = dict(zip("〇一二三四五六七八九", "0123456789"))
HNUM = set("〇零一二三四五六七八九十百千萬壹貳參拾")
def nk(c): return unicodedata.normalize("NFKC", c)


try:
    import hypua2jamo
except ImportError:  # python -m pip install hypua2jamo
    hypua2jamo = None
SIOT = {"ᄯ": "ᄄ", "ᄭ": "ᄁ", "ᄲ": "ᄈ", "ᄶ": "ᄍ", "ᄴ": "ᄊ"}


def modern(o):
    """한양 PUA 옛한글 중 된시옷(ㅼ·ㅺ·ㅽ·ㅾ) 음절 → 현대 된소리 음절. 아니면 None.
    규칙(10-06): 된시옷은 현대 된소리로 적는다(한국사DB 입력문·k1930 라벨과 같은 관례)."""
    if hypua2jamo is None or not (0xE000 <= ord(o) <= 0xF8FF):
        return None
    j = hypua2jamo.translate(o)
    if not j or j[0] not in SIOT:
        return None
    s = unicodedata.normalize("NFC", SIOT[j[0]] + j[1:])
    return s if len(s) == 1 and is_hang(s) else None


# ---------- 입력문 ----------
def ref_seq():
    """기사 입력문 → [(글자, 기사id, 쪽)] — 대조 글자만. ＜n＞은 그 앞 글자들이 n쪽임을 뜻한다."""
    arts = json.loads((W / "articles.json").read_text(encoding="utf-8"))
    seq = []
    for a in arts:
        t = a["text"]
        start = len(seq)
        for part in re.split(r"(＜\d+＞|<\d+>)", t):
            m = re.fullmatch(r"[＜<](\d+)[＞>]", part)
            if m:
                pg = int(m.group(1))
                for k in range(start, len(seq)):
                    if seq[k][2] is None:
                        seq[k] = (seq[k][0], seq[k][1], pg)
                continue
            for c in part:
                if is_key(c):
                    seq.append((c, a["id"], None))
    return seq


# ---------- 정렬 ----------
def cost(o, r):
    """0=같음, 0.2=표기 차이 후보(이체자·한자숫자↔아라비아숫자·한자↔독음 한글·된시옷), 1=다름."""
    if o == r or nk(o) == nk(r) or o.lower() == r.lower():
        return 0.0
    if NUM.get(o) == r or (o in HNUM and r.isdigit()):
        return 0.2
    if is_han(o) and (is_han(r) and readings(o) & readings(r) or is_hang(r) and norm(r) in readings(o)):
        return 0.2
    if modern(o) == r:
        return 0.2
    return 1.0


def variant(o, r, p):
    """cost 0.2 자리 → (고침, 근거). 입력문이 원본을 바꿔 적은 경우이므로 원본 쪽(OCR)을 따르되 확신 낮으면 보류."""
    if modern(o) == r:
        return r, "된시옷_현대표기"
    if o in HNUM and r.isdigit():
        return ("KEEP", "숫자표기") if p >= 0.9 else (None, "숫자표기_확신낮음")
    if is_hang(r):
        return ("KEEP", "입력문_한글로씀") if p >= 0.9 else (None, "입력문_한글로씀_확신낮음")
    return ("KEEP", "이체자") if p >= 0.9 else (None, "이체자_확신낮음")


def align(ok, ref, lo, hi):
    """ok를 ref[lo:hi]에 반전역 정렬(ref 앞뒤 공짜). 반환 (비용, ops[(i|None, j|None, c)])."""
    R = ref[lo:hi]
    n, m = len(ok), len(R)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    B = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i; B[i][0] = 1
    for i in range(1, n + 1):
        oi = ok[i - 1]; Di = D[i]; Dp = D[i - 1]; Bi = B[i]
        for j in range(1, m + 1):
            a, b, d = Dp[j - 1] + cost(oi, R[j - 1][0]), Dp[j] + 1, Di[j - 1] + 1
            if a <= b and a <= d:
                Di[j] = a; Bi[j] = 0
            elif b <= d:
                Di[j] = b; Bi[j] = 1
            else:
                Di[j] = d; Bi[j] = 2
    j = min(range(m + 1), key=lambda x: D[n][x])
    best, i, ops = D[n][j], n, []
    while i > 0:
        if j == 0 or B[i][j] == 1:
            ops.append((i - 1, None, 1.0)); i -= 1
        elif B[i][j] == 0:
            ops.append((i - 1, lo + j - 1, cost(ok[i - 1], R[j - 1][0]))); i -= 1; j -= 1
        else:
            ops.append((None, lo + j - 1, 1.0)); j -= 1
    ops.reverse()
    while ops and ops[0][0] is None:
        ops.pop(0)
    return best, ops


# ---------- 판정 ----------
def near_digit(ops, t, ref):
    """ops[t] 양옆(2칸 안)에 입력문 아라비아 숫자가 있나."""
    return any(x[1] is not None and ref[x[1]][0].isdigit() for x in ops[max(0, t - 2):t + 3])


def near_hnum(ops, t, ocr, keys):
    """ops[t] 양옆(2칸 안)에 OCR 한자 숫자가 있나."""
    return any(x[0] is not None and ocr[keys[x[0]]] in HNUM for x in ops[max(0, t - 2):t + 3])


def judge(line, ops, ref):
    ocr = line["text"]
    keys = [k for k, c in enumerate(ocr) if is_key(c)]
    prob = line["prob"] + [1.0] * (len(ocr) - len(line["prob"]))
    lab, add_after, diffs = list(ocr), defaultdict(str), []
    for t, (i, j, c) in enumerate(ops):
        if c == 0 and i is not None and j is not None:
            continue
        if i is not None and j is not None:
            k = keys[i]; o, r = ocr[k], ref[j][0]
            if c == 0.2:
                fix, why = variant(o, r, prob[k])
            elif is_han(o) and is_han(r) and prob[k] >= 0.97:
                fix, why = None, "한자_입력문과다름_OCR확신"   # 입력문 오타(晝夜→畵夜)일 수 있다
            else:
                fix, why = r, "입력문"
            diffs.append(dict(t="sub", k=k, i=i, o=o, r=r, p=prob[k], fix=fix, why=why))
        elif i is not None:
            k = keys[i]
            if ocr[k] in HNUM and prob[k] >= 0.9 and near_digit(ops, t, ref):
                fix, why = "KEEP", "숫자표기"      # 一千九百二十 ↔ 1920 처럼 자릿수가 다름
            else:
                fix, why = ("DEL", "OCR에만_확신낮음") if prob[k] < 0.5 else (None, "OCR에만")
            diffs.append(dict(t="ins", k=k, i=i, o=ocr[k], p=prob[k], fix=fix, why=why))
        else:
            prev = next((x for x in ops[t::-1] if x[0] is not None), None)
            nxt = next((x for x in ops[t:] if x[0] is not None), None)
            kp = keys[prev[0]] if prev else -1
            inner = prev is not None and nxt is not None
            if ref[j][0].isdigit() and near_hnum(ops, t, ocr, keys):
                fix, why = "KEEP", "숫자표기"      # 十 ↔ 10
            else:
                fix, why = ("ADD" if inner else None), "입력문에만"
            diffs.append(dict(t="del", k=kp, i=(prev[0] + 0.5) if prev else -0.5, r=ref[j][0], fix=fix, why=why))
    adds = [d for d in diffs if d["fix"] == "ADD"]
    if len(adds) > 1:
        for d in adds:
            d["fix"], d["why"] = None, "누락여럿"
    for d in diffs:
        if d["fix"] in (None, "KEEP"):
            continue
        if d["fix"] == "DEL":
            lab[d["k"]] = ""
        elif d["fix"] == "ADD":
            add_after[d["k"]] += d["r"]
        else:
            lab[d["k"]] = d["fix"]
    label = add_after.get(-1, "") + "".join(ch + add_after.get(k, "") for k, ch in enumerate(lab))
    # 문장부호(대조 안 됨): 확신 낮으면 보류
    for k, ch in enumerate(ocr):
        if not is_key(ch) and not ch.isspace() and prob[k] < 0.7:
            diffs.append(dict(t="punct", k=k, o=ch, p=prob[k], fix=None, why="부호_확신낮음"))
    # 점·줄표 반복(대조 안 됨): 줄 길이(글자 칸 수)에 비해 너무 길면 OCR 폭주로 보고 보류(dot_filter.py와 같은 기준, 10-08)
    x0, y0, x1, y1 = line["box"]
    cells = (y1 - y0) / max(1, x1 - x0) if line["dir"] == "v" else (x1 - x0) / max(1, y1 - y0)
    if len(ocr) >= 100 or (re.search(r"([·…‥.\-―—─])\1{7,}", ocr) and len(ocr) / max(1e-6, cells) > 2.0):
        diffs.append(dict(t="line", fix=None, why="점선_폭주"))
    changed = sum(1 for d in diffs if d["fix"] not in (None, "KEEP") and d["why"] != "된시옷_현대표기")
    unresolved = sum(1 for d in diffs if d["fix"] is None)
    if changed > 2 or changed > 0.10 * max(1, len(keys)):
        unresolved += 1
        diffs.append(dict(t="line", fix=None, why=f"고침많음{changed}"))
    st = "review" if unresolved else "ref_fixed" if label != ocr else "ref_match"
    return st, label, diffs


def run():
    ref = ref_seq()
    K = [nk(c) for c, _, _ in ref]
    idx = defaultdict(list)
    for p in range(len(K) - 2):
        idx["".join(K[p:p + 3])].append(p)
    rows, cnt = [], Counter()
    for f in sorted((W / "ocr").glob("*.json")):
        pg = json.loads(f.read_text(encoding="utf-8"))
        last = None
        for li, line in enumerate(pg["lines"]):
            ok = [c for c in line["text"] if is_key(c)]
            row = dict(page=pg["page"], line=li, box=line["box"], dir=line["dir"], kind=line["kind"], ocr=line["text"])
            best = None
            if len(ok) >= 4:
                Kq = [nk(c) for c in ok]
                votes = Counter()
                for p in range(len(Kq) - 2):
                    for q in idx.get("".join(Kq[p:p + 3]), ())[:200]:
                        votes[(q - p) // 8] += 1
                for off8, v in votes.most_common(3):
                    if v < 2:
                        continue
                    lo = max(0, off8 * 8 - 12); hi = min(len(ref), off8 * 8 + len(ok) + 20)
                    c, ops = align(ok, ref, lo, hi)
                    if best is None or c < best[0]:
                        best = (c, ops)
                if best is not None and best[0] > 0.5 * len(ok):
                    best = None
            elif ok and last is not None:   # 짧은 줄: 앞줄 바로 뒤에서만
                c, ops = align(ok, ref, last, min(len(ref), last + len(ok) + 4))
                if c <= 0.34 * len(ok) and any(j is not None for _, j, _ in ops):
                    best = (c, ops)
            if best is None:
                st = "no_ref" if len(ok) >= 4 else "short"
                row.update(status=st, label=line["text"]); rows.append(row); cnt[st] += 1
                if st == "no_ref":
                    last = None
                continue
            c, ops = best
            js = [j for _, j, _ in ops if j is not None]
            st, label, diffs = judge(line, ops, ref)
            row.update(status=st, label=label, cost=round(c, 2), ref_span=[min(js), max(js) + 1],
                       article=ref[js[0]][1], ref_page=ref[js[0]][2],
                       ref_text="".join(ch for ch, _, _ in ref[min(js):max(js) + 1]), diffs=diffs)
            rows.append(row); cnt[st] += 1; last = max(js) + 1
    with open(W / "align.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    E = N = 0
    for r in rows:
        if r["status"] in ("ref_match", "ref_fixed"):
            E += ed(r["ocr"], r["label"]); N += len(r["label"])
    st = dict(lines=len(rows), status=dict(cnt), chars_checked=N, ocr_errors_fixed=E,
              cer_on_checked=round(E / max(1, N) * 100, 2),
              why=dict(Counter(d["why"] for r in rows for d in r.get("diffs", [])).most_common()))
    (W / "align_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1))


def ed(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


# ---------- 내보내기 ----------
GRADE = {"ref_match": "silver_ref", "ref_fixed": "silver_ref_fixed", "review": "review", "no_ref": "ocr_raw", "short": "ocr_raw"}


def export():
    import base64
    import io
    from PIL import Image
    out = lines_dir(NO)
    (out / "images").mkdir(parents=True, exist_ok=True)
    arts = {a["id"]: a for a in json.loads((W / "articles.json").read_text(encoding="utf-8"))}
    rows = [json.loads(l) for l in open(W / "align.jsonl", encoding="utf-8")]
    recs, cnt, cur = [], Counter(), (None, None)
    for r in rows:
        if cur[0] != r["page"]:
            cur = (r["page"], Image.open(W / "dsk" / f"{r['page']}.png").convert("L"))
        g = cur[1]
        x0, y0, x1, y1 = r["box"]
        lid = f"{TAG}{NO}_{r['page'].split('_')[-1]}_l{r['line']:03d}"
        g.crop((max(0, x0 - 2), max(0, y0 - 2), min(g.width, x1 + 2), min(g.height, y1 + 2))).save(out / "images" / f"{lid}.png")
        gr = GRADE[r["status"]]; cnt[gr] += 1
        a = arts.get(r.get("article"), {})
        recs.append(dict(id=lid, magazine=NAME, issue=f"{PFX}_{NO}", date=a.get("date"), article=r.get("article"),
                         title=a.get("title"), image_file=r["page"] + ".jpg", ref_page=r.get("ref_page"),
                         box=r["box"], dir=r["dir"], kind=r["kind"], image=f"images/{lid}.png",
                         label=r["label"].replace(" ", ""), ocr=r["ocr"], grade=gr, status=r["status"],
                         ref_text=r.get("ref_text"),
                         diffs=[{k: v for k, v in d.items() if k in ("t", "o", "r", "fix", "why", "p")} for d in r.get("diffs", [])],
                         train=gr in ("silver_ref", "silver_ref_fixed"),
                         rights="국사편찬위원회 한국사DB 원문이미지·입력문 — 이용조건 확인 전 비공개"))
    with open(out / "lines.jsonl", "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    tr = [x for x in recs if x["train"]]
    E = sum(ed(x["ocr"], x["label"]) for x in tr); N = sum(len(x["label"]) for x in tr)
    stats = dict(version=out.name, lines=len(recs), grades=dict(cnt), train_lines=len(tr), train_chars=N,
                 ocr_model="KMMOCR R4 = E10ep1, NDL 줄 분할", cer_checked_lines=round(E / max(1, N) * 100, 2),
                 source="https://db.history.go.kr/modern/level.do?levelId=" + PFX + "_" + NO,
                 test_set_overlap="없음(사상계 시험 20개 호와 무관한 잡지)",
                 scripts=["gb_fetch.py", "gb_ocr.py", "gb_align.py"])
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    items = []
    for x in recs:
        if x["grade"] != "review":
            continue
        b = io.BytesIO(); Image.open(out / x["image"]).save(b, "PNG")
        items.append(dict(id=x["id"], img=base64.b64encode(b.getvalue()).decode(), ocr=x["ocr"], label=x["label"],
                          ref=x["ref_text"] or "", why=" · ".join(sorted({d["why"] for d in x["diffs"] if d.get("fix") is None}))))
    sys.argv = [sys.argv[0]]
    import comet_export
    html = (comet_export.HTML.replace("__DATA__", json.dumps(items, ensure_ascii=False)).replace("__N__", str(len(items)))
            .replace("__TAG__", f"{TAG}{NO}").replace("__SET__", out.name)
            .replace("코메트 20호", f"{NAME} {NO}").replace("한글본", "입력문"))
    d = REVIEW; d.mkdir(exist_ok=True)
    (d / f"검수_{TAG}{NO}_보류.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    export() if "--export" in sys.argv else run()
