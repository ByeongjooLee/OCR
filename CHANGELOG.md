# CHANGELOG

## 2026-10-03 (이름)
- 프로젝트 이름을 **KMMOCR**(Korean Modern Magazine OCR)로 정함. 공개 모델 버전 R1 = 인식 모델 E5. 데이터 v1은 그대로.

## v1 (2026-10-03)
- `data/sasanggye_lines_v1`: 『사상계』 26개 호 308면, 세로줄 14,112개(학습 12,083 / 검증 2,029), 등급 silver.
- 기울기 보정 후 NDLOCR-Lite 줄 검출, 생성형 재판독 대조 라벨, 이체자 223쌍 정자 통일.
