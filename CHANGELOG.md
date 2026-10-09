# CHANGELOG

## v2 (2026-10-09)
- `data/gaebyeok_lines_v1`: 『개벽』(국사편찬위원회 한국사DB) 62개 호, 줄 388,142개(학습용 260,713) 메타데이터. 이미지는 `scripts/gb_restore.py`로 한국사DB 원문에서 복원한다. 시험 호 8개는 넣지 않았다.
- `scripts/`: 한국사DB 잡지 수집·OCR·대조·내보내기·복원 스크립트(gb_*.py, dot_filter.py), 잡지 코드는 `GB_MAG`로 고른다.

## 2026-10-03 (이름)
- 프로젝트 이름을 **KMMOCR**(Korean Modern Magazine OCR)로 정함. 공개 모델 버전 R1 = 인식 모델 E5. 데이터 v1은 그대로.

## v1 (2026-10-03)
- `data/sasanggye_lines_v1`: 『사상계』 26개 호 308면, 세로줄 14,112개(학습 12,083 / 검증 2,029), 등급 silver.
- 기울기 보정 후 NDLOCR-Lite 줄 검출, 생성형 재판독 대조 라벨, 이체자 223쌍 정자 통일.
