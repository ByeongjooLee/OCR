# -*- coding: utf-8 -*-
"""『개벽』(국사편찬위원회 한국사DB, levelId=ma_013) 호 단위 수집: 기사 입력문 + 원문 면 이미지.

  python -X utf8 작업도구/스크립트/gb_fetch.py 0010 [0020 ...]   (호 번호 = levelId 넷째 자리, 0010=제1호)

참고: github.com/Esantomi/gaebyeok-scraper(본문·메타만). 여기서는 면 이미지까지 받는다.
- 기사 목록: /modern/getChildItemLevelListAjax.do?parentId=ma_013_{호}&level=3
- 기사 본문: /modern/level.do?levelId=ma_013_{호}_{기사} 의 #cont_view. 원표기·한자 그대로, 면 경계 ＜n＞(n=원본 쪽 번호).
- 면 목록: /common/imageViewer.do?levelId=<기사 또는 호>, 원본: /common/imageProxy.do?filePath=/ma/013/{호}/ma_013_{호}_{면}.jpg
서버 부담을 줄이려 요청마다 쉰다(DELAY). 이미 받은 파일은 건너뛴다.
출력: ocr_data/gaebyeok_work/{호}/img/*.jpg, articles.json [{id,title,author,kind,date,text,images}]
"""
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gb_mag import WORK as OUT, PFX, MAG  # noqa: E402  (GB_MAG로 잡지 선택, 기본 개벽)
BASE = "https://db.history.go.kr"
DELAY = 1.5
UA = {"User-Agent": "Mozilla/5.0 (research; modernmag.kr OCR)"}


def get(u, binary=False):
    for k in range(4):
        try:
            r = urllib.request.urlopen(urllib.request.Request(BASE + u, headers=UA), timeout=60).read()
            time.sleep(DELAY)
            return r if binary else r.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            print("  retry", k, u, e, flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError(u)


def text_of(frag):
    frag = re.sub(r"<br\s*/?>", "\n", frag)
    frag = re.sub(r"</div>", "\n", frag)
    t = html.unescape(re.sub(r"<[^>]+>", "", frag))
    lines = [re.sub(r"[ \t ]+", " ", l).strip() for l in t.split("\n")]
    return "\n".join(l for l in lines if l)


def meta(t, name):
    m = re.search(r'<div class="tit">' + name + r'</div><div class="cont">(.*?)</div>', t, re.S)
    return text_of(m.group(1)) if m else ""


def article(aid):
    t = get(f"/modern/level.do?levelId={aid}")
    i = t.find('id="cont_view"')
    body = ""
    if i >= 0:
        j = t.find("</section>", i)
        body = text_of(t[t.find(">", i) + 1:j])
    return dict(id=aid, title=meta(t, "기사제목"), author=meta(t, "필자"), kind=meta(t, "기사형태"),
                date=meta(t, "발행일"), has_image="fnImageViewerPopup('" + aid in t, text=body)


def images(lid):
    t = get(f"/common/imageViewer.do?levelId={lid}")
    return sorted(set(re.findall(rf"ma/{MAG}/\d+/{PFX}_\d+_\d+\.jpg", t)))


def issue(no):
    d = OUT / no
    (d / "img").mkdir(parents=True, exist_ok=True)
    kids = get(f"/modern/getChildItemLevelListAjax.do?parentId={PFX}_{no}&level=3&sideNavYn=false")
    ids = list(dict.fromkeys(re.findall(r'data-id="(' + PFX + '_' + no + r'_\d+)"', kids)))
    af = d / "articles.json"
    old = {a["id"]: a for a in json.loads(af.read_text(encoding="utf-8"))} if af.exists() else {}
    arts = []
    for aid in ids:
        a = old.get(aid) or article(aid)
        if "images" not in a:
            a["images"] = images(aid) if a["has_image"] else []
        arts.append(a)
        print(aid, a["title"][:30], len(a["text"]), len(a["images"]), flush=True)
        af.write_text(json.dumps(arts, ensure_ascii=False, indent=1), encoding="utf-8")
    allimg = images(f"{PFX}_{no}")
    (d / "images.json").write_text(json.dumps(allimg, indent=0), encoding="utf-8")
    for p in allimg:
        f = d / "img" / Path(p).name
        if f.exists() and f.stat().st_size > 1000:
            continue
        f.write_bytes(get(f"/common/imageProxy.do?filePath=/{p}", binary=True))
    print(no, "articles", len(arts), "images", len(allimg), flush=True)


if __name__ == "__main__":
    for no in sys.argv[1:]:
        issue(no)
