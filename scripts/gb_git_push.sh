#!/usr/bin/env bash
# 공개 저장소에 쌓인 호 폴더를 커밋·push(호 단위로 나눠 1회 push가 2GB를 넘지 않게). 인자: 커밋 설명
cd "$(dirname "$0")/../../공개저장소_OCR" || exit 1
git add -A data scripts README.md CHANGELOG.md splits 2>/dev/null
git diff --cached --quiet && { echo "nothing to push"; exit 0; }
git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main 2>&1 | tail -2 && echo "pushed: $1 ($(git log --oneline -1 | cut -c1-7))"
