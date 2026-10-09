#!/usr/bin/env bash
# 한국사DB 원문제공 잡지 일괄: 수집 → OCR(R4) → 대조 → 내보내기 → 보정 면(dsk) 삭제(디스크 절약, img+deskew_deg로 재생성 가능).
#   GB_MAG=015 bash 작업도구/스크립트/gb_chain_mag.sh            (인자 없으면 사이트에서 호 목록을 받는다)
#   GB_MAG=015 bash 작업도구/스크립트/gb_chain_mag.sh 0010 0020
# 요청 간격은 gb_fetch.py DELAY(1.5초). 병렬로 돌리지 않는다(10-06 차단 경험).
cd "$(dirname "$0")/../.."
M=${GB_MAG:?GB_MAG 필요}
export GB_MAG=$M
W=$(python -X utf8 -c "import sys;sys.path.insert(0,'작업도구/스크립트');import gb_mag;print(gb_mag.WORK.as_posix())")
IDS="$*"
if [ -z "$IDS" ]; then
  IDS=$(curl -sL -m 60 -A "Mozilla/5.0" "https://db.history.go.kr/modern/getChildItemLevelListAjax.do?parentId=ma_$M&level=2&sideNavYn=false" | grep -o "data-id=\"ma_${M}_[0-9]\{4\}\"" | grep -o "[0-9]\{4\}\"" | tr -d '"' | awk '!s[$0]++')
fi
echo "ma_$M issues: $(echo $IDS | wc -w)"
for n in $IDS; do
  echo "=== $M $n $(date +%m-%d\ %H:%M)"
  L=$(python -X utf8 -c "import sys;sys.path.insert(0,'작업도구/스크립트');import gb_mag;print(gb_mag.lines_dir('$n').as_posix())")
  [ -f "$L/stats.json" ] && { echo "done already $n"; continue; }
  python -X utf8 작업도구/스크립트/gb_fetch.py $n > ocr_data/gbm${M}_fetch_$n.log 2>&1 || { echo "fetch FAIL $n"; tail -2 ocr_data/gbm${M}_fetch_$n.log; continue; }
  tail -1 ocr_data/gbm${M}_fetch_$n.log
  [ -z "$(ls $W/$n/img 2>/dev/null)" ] && { echo "no images $n"; continue; }
  작업도구/ocr_train/venv/Scripts/python.exe -X utf8 작업도구/스크립트/gb_ocr.py $n --device cuda > ocr_data/gbm${M}_ocr_$n.log 2>&1 || { echo "ocr FAIL $n"; continue; }
  python -X utf8 작업도구/스크립트/gb_align.py $n > /dev/null 2>&1 && python -X utf8 작업도구/스크립트/gb_align.py $n --export > ocr_data/gbm${M}_export_$n.log 2>&1 || { echo "align FAIL $n"; continue; }
  rm -rf "$W/$n/dsk"
  python -X utf8 -c "import json;s=json.load(open('$L/stats.json',encoding='utf-8'));print('$n', s['grades'], 'train', s['train_lines'], 'CER', s['cer_checked_lines'])"
  echo "disk free: $(df -h /c | tail -1 | awk '{print $4}')"
done
echo "=== ALL DONE ma_$M $(date +%m-%d\ %H:%M)"
