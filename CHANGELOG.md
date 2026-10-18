# CHANGELOG

## v2.2 (2026-10-18)
- `data/sasanggye_lines_v2`: 『사상계』 26개 호 308면, 세로줄 14,091개(학습 12,064 / 검증 2,027). v1 라벨에 KMMOCR 대조 재교정(면 이미지로 확인한 판정)을 얹었습니다. v1과 라벨이 다른 줄 198개(`label_prev`), 등급 `silver` 13,886 / `silver_fixed` 205. 호별 폴더·WebP·KMMOCR 인식 결과(`ocr`, `ocr_alt`) 형식. v1 폴더는 그대로 둡니다.
- `data/sasanggye_kmmocr_corr_v1`: KMMOCR 『사상계』 기사 판독에서 이미지와 대조해 고친 줄만 215개(6개 호, 등급 `bronze_kmmocr_corr`).
- `scripts/`: `sg318_export_recorr.py`(재교정 판정을 줄 라벨에 얹기), `sg_kmmocr_corr_export.py`(고친 줄 내보내기), `release_build.py`(호별 WebP 형식·인식 결과 옵션).

## v2.1 (2026-10-10~)
- 『별건곤』·『동광』·『삼천리』(국사편찬위원회 한국사DB): 호별 폴더 `data/ma0NN_lines_v1/<호>/`에 학습 줄 이미지(WebP)와 `lines.jsonl`(라벨·KMMOCR R4 인식 결과·판정)을 20개 호씩 올림. 『개벽』은 v2의 메타데이터만 둔다.

## v2 (2026-10-09)
- `data/gaebyeok_lines_v1`: 『개벽』(국사편찬위원회 한국사DB) 62개 호, 줄 388,142개(학습용 260,713) 메타데이터. 이미지는 `scripts/gb_restore.py`로 한국사DB 원문에서 복원한다. 시험 호 8개는 넣지 않았다.
- `scripts/`: 한국사DB 잡지 수집·OCR·대조·내보내기·복원 스크립트(gb_*.py, dot_filter.py), 잡지 코드는 `GB_MAG`로 고른다.

## 2026-10-03 (이름)
- 프로젝트 이름을 **KMMOCR**(Korean Modern Magazine OCR)로 정함. 공개 모델 버전 R1 = 인식 모델 E5. 데이터 v1은 그대로.

## v1 (2026-10-03)
- `data/sasanggye_lines_v1`: 『사상계』 26개 호 308면, 세로줄 14,112개(학습 12,083 / 검증 2,029), 등급 silver.
- 기울기 보정 후 NDLOCR-Lite 줄 검출, 생성형 재판독 대조 라벨, 이체자 223쌍 정자 통일.
