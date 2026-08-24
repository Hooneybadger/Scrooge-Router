<!--
SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
SPDX-License-Identifier: Apache-2.0
-->

# 변경 이력

[Keep a Changelog](https://keepachangelog.com/ko/1.1.0/) 형식을 따르고
버전은 [유의적 버전](https://semver.org/lang/ko/)을 따릅니다.

## [미출시]

### 추가

- 이슈와 PR 템플릿, 기여 안내를 갖췄습니다.
- 테스트와 정적 검사, 라이선스 검사를 실행하는 CI를 추가했습니다.
- Premium에 예측 예산 브레이크를 두어, 부모가 `ax31`을 고른 문항 중
  이득이 기대되는 것만 `axk1-think`로 올립니다. Fast와 Balanced는 계열
  가드와 같습니다. 계열 가드 모듈과 아티팩트는 롤백으로 남겨 둡니다.

### 변경

- 제출 진입점을 `budget_brake_router`로 바꿨습니다.
- 잔여 묶음이 0.75 이상인 Premium 배치에는 부모 가드와 잔여 K1 제외를
  적용하고, 한 묶음이 0.75 이상인 Fast 배치에는 예측 상한 1.07을
  적용합니다. 공개 Dev 점수 `0.669517`은 그대로입니다.
- Premium 예측 브레이크를 3.25에서 3.80으로 올렸습니다. Fast와 Balanced
  선택은 그대로이고, 공개 Dev 점수는 `0.670710`입니다.
