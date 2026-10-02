# -*- coding: utf-8 -*-
"""318면 라벨 정제: 기존 전사 ↔ Fable T 재판독 불일치 → 유형·NDL 보조의견·이미지 위치를 붙인 검수 목록.

  python -X utf8 작업도구/스크립트/sg318_triage.py

- 정렬: 기존 전사(T)를 기준으로 Fable(F)을 반전역 정렬(F의 앞뒤 초과분은 무비용) → 범위 밖 텍스트 무시.
- NDL 보조의견: NDLOCR-Lite 줄 텍스트를 T에 정렬(sg318_build.align_page)해 같은 자리의 NDL 글자를 붙인다.
  NDL은 한글을 읽지 못하므로 한자 자리에서만 의미가 있다. 정답 판정이 아니라 검수 순서용.
- 범위 불일치 면(반전역 불일치율 > 15%)은 목록에서 빼고 따로 기록한다.
출력: 학습데이터_사상계318/검수/review_queue.jsonl, crops/, triage_summary.json, scope_pages.json
"""
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sg318_build import OUT as D, align_page, clean_transcript, load_manifest  # noqa: E402

R = D / "검수"
(R / "crops").mkdir(parents=True, exist_ok=True)


def kind(ch):
    o = ord(ch)
    if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "한글"
    if ch == "〇" or unicodedata.name(ch, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
        return "한자"
    return None


def semi_align(a, b):
    """a 전체를 b의 부분에 맞춘다(b 앞뒤는 무비용). ops: (op,i,j)."""
    n, m = len(a), len(b)
    Dm = np.zeros((n + 1, m + 1), dtype=np.int32)
    bb = np.array(list(b)) if m else np.array([], dtype="<U1")
    cols = np.arange(m + 1)
    for i in range(1, n + 1):
        sub = (bb != a[i - 1]).astype(np.int32)
        row = np.concatenate(([Dm[i - 1, 0] + 1], np.minimum(Dm[i - 1, 1:] + 1, Dm[i - 1, :-1] + sub)))
        Dm[i] = np.minimum.accumulate(row - cols) + cols
    j = int(Dm[n].argmin()); i = n; ops = []
    while i > 0:
        if j and Dm[i, j] == Dm[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("=" if a[i - 1] == b[j - 1] else "S", i - 1, j - 1)); i -= 1; j -= 1
        elif Dm[i, j] == Dm[i - 1, j] + 1:
            ops.append(("D", i - 1, None)); i -= 1
        else:
            ops.append(("I", None, j - 1)); j -= 1
    return ops[::-1], int(Dm[n].min())


# 한글 자모 분해: 옛 표기·된소리 표기 차이 판정용
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
TENSE = {frozenset(p) for p in [("ㄱ", "ㄲ"), ("ㄷ", "ㄸ"), ("ㅂ", "ㅃ"), ("ㅅ", "ㅆ"), ("ㅈ", "ㅉ"), ("ㅅ", "ㅆ")]}


def jamo(c):
    o = ord(c) - 0xAC00
    if not 0 <= o < 11172:
        return None
    return CHO[o // 588], JUNG[(o % 588) // 28], JONG[o % 28]


def is_orth(old, new):
    """옛 표기/현대 표기 차이로 보이는 한 글자 치환: 받침 ㅅ↔ㅆ(겟/겠·엇/었·잇/있), 초성 된소리 표기(싻/싹 계열),
    읍↔습. 인쇄 원문이 옛 표기인지 모델이 현대화했는지는 이미지로만 판정할 수 있다."""
    if len(old) != 1 or len(new) != 1:
        return False
    if {old, new} == {"읍", "습"}:
        return True
    a, b = jamo(old), jamo(new)
    if not a or not b:
        return False
    diff = [k for k in range(3) if a[k] != b[k]]
    if len(diff) != 1:
        return False
    k = diff[0]
    if k == 2:
        return frozenset((a[2], b[2])) in TENSE or {a[2], b[2]} <= {"ㅅ", "ㅆ", "ㄳ", "ㄺ"}
    if k == 0:
        return frozenset((a[0], b[0])) in TENSE
    return False


# 같은 글자의 자형 이체(정자·속자·일본 신자체 등). 뜻·용법이 갈릴 수 있는 쌍(峰/峯, 摸/模, 辯/辨, 鐘/鍾, 像/象 등)은
# 넣지 않는다. 앞이 한국 인쇄에서 흔한 정자 쪽이다.
VARIANT_PAIRS = """硏研 槪概 絕絶 產産 淸清 旣既 强強 敍叙 冊册 緖緒 戲戱 畫畵 卽即 靑青 値值 尙尚 效効 昻昂 默黙
說説 爲為 敎教 內内 晩晚 對対 國国 學学 會会 體体 來来 錄録 悅悦 脫脱 銳鋭 閱閲 歷歴 曆暦 鄕郷 黑黒 吳呉 綠緑
稅税 爭争 眞真 步歩 涉渉 德徳 惠恵 揭掲 彥彦 戶户 拔抜 姬姫 溫温 黃黄 廣広 擧挙 圖図 價価 據拠 稱称 隱隠 從従
證証 單単 寫写 讀読 變変 藏蔵 狀状 將将 壯壮 莊荘 帶帯 驛駅 淚涙 經経 繼継 續続 總総 縣県 樂楽 藥薬 數数 實実
聲声 齒歯 氣気 處処 發発 廢廃 鬪闘 戰戦 傳伝 轉転 舊旧 歸帰 參参 惱悩 腦脳 亂乱 辭辞 齊斉 濟済 劑剤 兩両 滿満
蠻蛮 灣湾 營営 勞労 榮栄 櫻桜 獨独 觸触 屬属 圓円 團団 圍囲 國圀 豫予 與与 譽誉 爐炉 盡尽 晝昼 觀観 權権 歡歓
勸勧 戀恋 蠶蚕 鑛鉱 擴拡 黨党 當当 嘗甞 稻稲 豐豊 禮礼 靈霊 鹽塩 醫医 壓圧 惡悪 亞亜 假仮 價価 儉倹 劍剣 險険
驗験 檢検 縱縦 澤沢 擇択 譯訳 驛駅 釋釈 拂払 佛仏 沸沸 竝並 邊辺 遲遅 隨随 髓髄 碎砕 粹粋 醉酔 雜雑 錢銭 踐践
淺浅 殘残 棧桟 雙双 賣売 讀読 續続 寶宝 搜捜 插挿 收収 敍叙 恆恒 冰氷 姊姉 卷巻 圈圏 勳勲 薰薫 戾戻 龜亀 鷄鶏
溪渓 契契 狹狭 峽峡 俠侠 突突 器噐 臭臭 類類 嘆歎 謠謡 搖揺 遙遥 隣鄰 曉暁 燒焼 卑卑 碑碑 祕秘 祖祖 神神 社社
福福 祝祝 禍禍 視視 都都 著著 署署 者者 緣縁 黃黄
峰峯 拋抛 涼凉 倂併 顔顏 館舘 靭靱 脈脉 揷挿 每毎 飜翻 闊濶 糾糺 竊窃 簒篡 羈覊 晉晋 裏裡 愼慎 晳晰 晳晣""".split()
# 2026-10-02 추가: 사용자 지적(拋/抛)으로 검수 목록의 한 글자 쌍 1,571종을 전수 확인해 자형 이체만 더했다.
# 峰/峯은 처음에 제외했으나 같은 글자의 자형 차이라 이체자로 옮겼다.
VARIANT = {}
for p in VARIANT_PAIRS:
    if len(p) == 2 and p[0] != p[1]:
        VARIANT[p[0]] = p[0]; VARIANT[p[1]] = p[0]


def fold(s):
    return "".join(VARIANT.get(c, c) for c in s)


def category(old, new):
    if old and new and old != new and fold(old) == fold(new):
        return "이체자"
    s = old + new
    # ○(도형)와 〇(한자 영)은 인쇄상 구별되지 않는 부호화 차이 → 검수 대상 아님(정규화 규칙으로 처리)
    if s and set(s) <= {"○", "〇"}:
        return "부호"
    if not any(kind(c) for c in s):
        return "부호"
    if (not old or not new) and len(old) + len(new) >= 8:
        return "범위"
    if any(kind(c) == "한자" for c in s):
        return "한자"
    if is_orth(old, new):
        return "옛표기"
    return "한글"


PRIORITY = {("한자", "NDL=Fable"): 1, ("한자", "NDL없음"): 2, ("한자", "NDL=기존"): 3,
            ("한글", None): 4, ("옛표기", None): 5, ("이체자", None): 6, ("범위", None): 8, ("부호", None): 9}


def main():
    queue, scope_pages, stats = [], [], Counter()
    lines_by_page = {}
    for l in open(D / "lines.jsonl", encoding="utf-8"):
        x = json.loads(l)
        lines_by_page.setdefault(x["page_id"], []).append(x)
    for r in load_manifest():
        fp = D / "fable_T" / f"{r['id']}.txt"
        jp = D / "ndl" / f"{r['id']}.json"
        if not fp.exists() or not jp.exists():
            continue
        T, _ = clean_transcript(r["transcript"])
        F, f_unc = clean_transcript(fp.read_text(encoding="utf-8"))
        ops, e = semi_align(T, F)
        rate = e / max(1, len(T))
        if rate > 0.15:
            scope_pages.append(dict(page=r["id"], semi_rate=round(rate, 4), N=len(T), N_fable=len(F)))
            continue
        ndl_lines = [b for blk in json.loads(jp.read_text(encoding="utf-8"))["contents"] for b in blk]
        res = align_page(T, ndl_lines)
        t2h, H = (res[0], res[2]) if res else ([None] * len(T), [])
        plines = lines_by_page.get(r["id"], [])
        img = None
        k = 0
        while k < len(ops):
            if ops[k][0] == "=":
                k += 1; continue
            s = k
            while k < len(ops) and ops[k][0] != "=":
                k += 1
            seg = ops[s:k]
            ti = [i for _, i, _ in seg if i is not None]
            fj = [j for _, _, j in seg if j is not None]
            pos = ti[0] if ti else next((o[1] + 1 for o in reversed(ops[:s]) if o[1] is not None), 0)
            old = "".join(T[i] for i in ti)
            new = "".join(F[j] for j in fj)
            cat = category(old, new)
            vote = None
            if cat == "한자":
                hs = [t2h[i] for i in ti if t2h[i] is not None]
                ndl = "".join(H[h] for h in hs) if hs else ""
                vote = "NDL=Fable" if ndl and ndl == new else ("NDL=기존" if ndl and ndl == old else "NDL없음")
            else:
                ndl = None
            pr = PRIORITY.get((cat, vote), PRIORITY.get((cat, None), 9))
            stats[(cat, vote)] += 1
            if pr >= 8:
                continue
            # 이미지 위치: tspan에 pos가 드는 줄
            ln = next((x for x in plines if x["tspan"][0] <= pos <= x["tspan"][1]), None)
            crop_name = None
            if ln:
                if img is None:
                    img = Image.open(D / "pages" / f"{r['id']}.png").convert("L")
                x0, y0, x1, y1 = ln["box"]
                a, b = ln["tspan"]
                frac = (pos - a + 0.5) / max(1, b - a + 1)
                if ln["vertical"]:
                    cy = y0 + frac * (y1 - y0); w = x1 - x0
                    box = (max(0, x0 - int(1.6 * w)), max(0, int(cy - 4.5 * w)), min(img.width, x1 + int(1.6 * w)), min(img.height, int(cy + 4.5 * w)))
                    mark = (x0 - box[0], int(cy - box[1] - 0.7 * w), x1 - box[0], int(cy - box[1] + 0.7 * w))
                else:
                    cx = x0 + frac * (x1 - x0); h = y1 - y0
                    box = (max(0, int(cx - 6 * h)), max(0, y0 - h), min(img.width, int(cx + 6 * h)), min(img.height, y1 + h))
                    mark = (int(cx - box[0] - 0.8 * h), y0 - box[1], int(cx - box[0] + 0.8 * h), y1 - box[1])
                c = img.crop(box).convert("RGB")
                from PIL import ImageDraw
                ImageDraw.Draw(c).rectangle(mark, outline=(220, 30, 30), width=2)
                c = c.resize((c.width * 3, c.height * 3), Image.LANCZOS)
                crop_name = f"{r['id']}_{pos:05d}.png"
                c.save(R / "crops" / crop_name)
            queue.append(dict(id=f"{r['id']}:{pos}", page=r["id"], split=r["split"], pos=pos, old=old, fable=new,
                              ndl=ndl, vote=vote, category=cat, priority=pr,
                              fable_uncertain=any(j in f_unc for j in fj) or any((fj and j == fj[0] - 1) for j in f_unc),
                              ctx_before=T[max(0, pos - 12):pos], ctx_after=T[pos + len(old):pos + len(old) + 12],
                              line=ln["id"] if ln else None, crop=f"crops/{crop_name}" if crop_name else None))
    queue.sort(key=lambda q: (q["priority"], q["page"], q["pos"]))
    with open(R / "review_queue.jsonl", "w", encoding="utf-8") as w:
        for q in queue:
            w.write(json.dumps(q, ensure_ascii=False) + "\n")
    (R / "scope_pages.json").write_text(json.dumps(scope_pages, ensure_ascii=False, indent=1), encoding="utf-8")
    summ = dict(pages_compared=len({q["page"] for q in queue}) , scope_excluded_pages=len(scope_pages),
                queue=len(queue), by_priority=dict(sorted(Counter(q["priority"] for q in queue).items())),
                by_category_vote={f"{c}|{v}": n for (c, v), n in sorted(stats.items(), key=lambda x: str(x))},
                fable_uncertain_in_queue=sum(q["fable_uncertain"] for q in queue),
                no_crop=sum(q["crop"] is None for q in queue))
    (R / "triage_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
