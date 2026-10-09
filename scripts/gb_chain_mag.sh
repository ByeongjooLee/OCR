#!/usr/bin/env bash
# 한국사DB 원문제공 잡지 일괄: 수집 → OCR(R4) → 대조 → 내보내기 → 보정 면(dsk) 삭제 → 학습 줄 WebP+lines.jsonl을 공개 저장소 폴더로(gb_publish_imgs) → 면·줄 PNG 삭제(시험 호 제외) → 20개 호마다 push.
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
PUB=0; BATCH=""
for n in $IDS; do
  echo "=== $M $n $(date +%m-%d\ %H:%M)"
  L=$(python -X utf8 -c "import sys;sys.path.insert(0,'작업도구/스크립트');import gb_mag;print(gb_mag.lines_dir('$n').as_posix())")
  if [ -f "$L/stats.json" ]; then   # 처리는 끝났고 공개 폴더가 없으면 내보내기만
    NM=$(python -X utf8 -c "import sys;sys.path.insert(0,'작업도구/스크립트');import gb_mag;print('ma'+gb_mag.MAG+'_lines_v1')")
    if [ ! -f "공개저장소_OCR/data/$NM/$n/lines.jsonl" ] && [ -d "$L/images" ]; then
      python -X utf8 작업도구/스크립트/gb_publish_imgs.py 공개저장소_OCR/data $n
      if ! python -X utf8 -c "import sys;sys.path.insert(0,'작업도구/스크립트');import gb_mag;sys.exit(0 if gb_mag.is_test('$n') else 1)"; then rm -rf "$W/$n/img" "$L/images"; fi
      PUB=$((PUB+1)); BATCH="$BATCH $n"
    fi
    echo "done already $n"; continue
  fi
  python -X utf8 작업도구/스크립트/gb_fetch.py $n > ocr_data/gbm${M}_fetch_$n.log 2>&1 || { echo "fetch FAIL $n"; tail -2 ocr_data/gbm${M}_fetch_$n.log; continue; }
  tail -1 ocr_data/gbm${M}_fetch_$n.log
  [ -z "$(ls $W/$n/img 2>/dev/null)" ] && { echo "no images $n"; continue; }
  작업도구/ocr_train/venv/Scripts/python.exe -X utf8 작업도구/스크립트/gb_ocr.py $n --device cuda > ocr_data/gbm${M}_ocr_$n.log 2>&1 || { echo "ocr FAIL $n"; continue; }
  python -X utf8 작업도구/스크립트/gb_align.py $n > /dev/null 2>&1 && python -X utf8 작업도구/스크립트/gb_align.py $n --export > ocr_data/gbm${M}_export_$n.log 2>&1 || { echo "align FAIL $n"; continue; }
  rm -rf "$W/$n/dsk"
  python -X utf8 작업도구/스크립트/gb_publish_imgs.py 공개저장소_OCR/data $n
  if ! python -X utf8 -c "import sys;sys.path.insert(0,'작업도구/스크립트');import gb_mag;sys.exit(0 if gb_mag.is_test('$n') else 1)"; then rm -rf "$W/$n/img" "$L/images"; fi
  PUB=$((PUB+1)); BATCH="$BATCH $n"
  if [ $PUB -ge 20 ]; then bash 작업도구/스크립트/gb_git_push.sh "$(python -X utf8 -c "import sys;sys.path.insert(0,'작업도구/스크립트');import gb_mag;print(gb_mag.NAME)") 줄 이미지+라벨:$BATCH"; PUB=0; BATCH=""; fi
  python -X utf8 -c "import json;s=json.load(open('$L/stats.json',encoding='utf-8'));print('$n', s['grades'], 'train', s['train_lines'], 'CER', s['cer_checked_lines'])"
  echo "disk free: $(df -h /c | tail -1 | awk '{print $4}')"
done
GB_MAG=$M python -X utf8 작업도구/스크립트/gb_publish.py 공개저장소_OCR/data > /dev/null
bash 작업도구/스크립트/gb_git_push.sh "$(python -X utf8 -c "import sys;sys.path.insert(0,'작업도구/스크립트');import gb_mag;print(gb_mag.NAME)") 줄 이미지+라벨:$BATCH + 메타데이터"
echo "=== ALL DONE ma_$M $(date +%m-%d\ %H:%M)"
