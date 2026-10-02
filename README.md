# 1950년대 『사상계』 국한문 세로쓰기 OCR 학습데이터

1950년대 잡지 『사상계』의 국한문 혼용 세로쓰기 활판 인쇄면에서 잘라낸 **줄 이미지와 판독 라벨**입니다. 생성형 AI 없이 로컬에서 재현할 수 있는 OCR 모델을 학습하려고 만들었습니다. 데이터는 계속 쌓아 버전을 올립니다.

Line-level OCR training data (cropped line images + transcriptions) from the 1950s Korean magazine *Sasanggye* (사상계), printed in vertical mixed Hangul–Hanja script. Research use only (CC BY-NC 4.0).

## 구성

| 경로 | 내용 |
|---|---|
| `data/sasanggye_lines_v1/` | 줄 이미지 14,112개(학습 12,083 / 검증 2,029), `lines.jsonl`, `charset.txt`, `stats.json` |
| `rules/variant_folding_v1.json` | 이체자 → 정자 통일 규칙 223쌍 |
| `splits/heldout_test_issues.json` | 평가용으로 떼어 둔 호 목록(이 호들은 데이터에 없음) |
| `layout/` | 『사상계』·『조광』·『문장』·『가톨릭청년』 판형 통계와 단 템플릿 |
| `scripts/` | 줄 분할 정렬, 기울기 보정, 내보내기, 문자 언어모델, 채점 스크립트 |

## `lines.jsonl` 필드

| 필드 | 뜻 |
|---|---|
| `id` | 줄 ID (`sg_<호>_p<면>_L<줄>`) |
| `split` | `train` / `val` (호 단위로 나눔) |
| `image` | 줄 이미지 경로. 세로줄은 세로로 선 원래 방향 그대로이고, 학습할 때 반시계 90° 돌려 눕힌다 |
| `label` | 판독 라벨(NFC, 이체자는 정자로 통일, 옛 표기와 원문 오식은 그대로) |
| `magazine`, `issue_id`, `page_id`, `page_no`, `article` | 출처 |
| `box` | 기울기 보정한 면에서의 줄 상자 `[x0, y0, x1, y1]` |
| `deskew_deg` | 면 기울기 보정 각도 |
| `grade` | 라벨 등급 (아래) |
| `rights` | 권리 상태 참고값 (`unknown`, `pd`, `protected`) |

### 라벨 등급

| 등급 | 뜻 |
|---|---|
| `gold` | 사람이 이미지와 대조해 확정 |
| `silver` | 생성형 모델 판독 두 벌을 대조하고 불일치는 규칙으로 정함 (v1 전부) |
| `bronze` | 생성형 단일 판독 |
| `ocr_raw` | 기성 OCR 출력 |
| `pseudo` | 우리 모델의 자가학습 의사라벨 |

## 만든 방법 (v1)

1. 면 스캔을 기울기 보정한다(±3°, 투영 분산 최대화).
2. [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite)로 줄을 검출한다.
3. 생성형 모델 재판독문을 검출한 줄에 정렬한다. 한자를 닻으로 써서 줄 경계를 정한다.
4. 가로줄, 정렬 품질이 낮은 줄(길이비 0.8~1.25 밖, 한자 닻 일치 60% 미만), 판독 불확실 글자가 든 줄, 61자 이상 줄은 뺀다.
5. 이체자를 정자로 통일한다(`rules/`).

자세한 수집·정제 과정과 한계는 [DATASHEET.md](DATASHEET.md)에 있습니다.

## 이용 조건

- **데이터**(`data/`, `layout/`, `rules/`): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 학술·비영리 연구에만 쓸 수 있습니다.
- **코드**(`scripts/`): MIT License (`LICENSE`).
- 면 전체 이미지는 배포하지 않습니다. 학습용으로 잘라낸 줄 단위 이미지와 라벨만 제공합니다.
- 원문 저작물의 권리는 각 저작권자에게 있습니다. 권리자께서 특정 글의 제외를 원하시면 이 저장소의 Issues로 알려 주십시오. 확인 후 다음 버전에서 해당 줄을 뺍니다.
- 제3자 구성요소의 출처는 [NOTICE.md](NOTICE.md)에 있습니다.

## 인용

논문 게재 후 인용 정보를 추가합니다.
