---
name: 로컬 코드 실행 금지
description: 현재 작업 환경은 서버가 아니라 로컬이므로, 코드 실행(python, torchrun 등)은 하지 않음
type: feedback
---

코드 실행(python, torchrun, pytest 등)은 서버에서만 해야 한다.

**Why:** 현재 로컬 환경에는 GPU/데이터/모델 weight 등이 없어 실행이 불가능하거나 의미 없음.

**How to apply:** Bash로 python 스크립트를 실행하려 할 때는 멈추고, 대신 코드 분석/계산으로 결과를 추론하거나 서버에서 실행할 명령어를 알려준다.
