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
- 품질·비용 분포를 트리로 예측하고 오목 접두 대기열로 배분하는
  `distributional_router`를 추가했습니다. 계열 가드 계층은 비교용으로 남겨 둡니다.

### 변경

- 제출 진입점을 `distributional_router`로 바꿨습니다. 공개 Dev 점수는
  `0.715767`입니다. 세 등급 모두 한도의 95% 아래이고, Fast는 think
  모델을 쓰지 않습니다.
- 잔여 묶음이 0.75 이상인 Premium 배치에는 부모 가드와 잔여 K1 제외를
  적용하고, 한 묶음이 0.75 이상인 Fast 배치에는 예측 상한 1.07을
  적용합니다. 공개 Dev 점수 `0.669517`은 그대로입니다.
- Premium 예측 브레이크를 3.25에서 3.80으로 올렸습니다. Fast와 Balanced
  선택은 그대로이고, 공개 Dev 점수는 `0.670710`입니다.
- Premium 폭주 가드를 받은 배치 기준으로 바꿨습니다. 브레이크 블록에
  `runaway_share` 0.06을 실어 `min(runaway_absolute, runaway_share ×
  배치 예측 light)`를 씁니다. 기존 규칙보다 느슨해지지 않으므로 Train,
  Dev, 공개 2,640 배치의 Premium 선택과 Dev 점수 `0.670710`은 비트
  단위로 같고, 짧은 배치에서만 조입니다. 공개 Dev 스트레스 뷰 5,840개의
  한도 초과가 55건에서 0건으로, 최악 실현 비율이 5.1861에서 3.3314로
  내려갔습니다.

### 수정

- Premium 폭주 가드가 받은 배치가 아니라 Train 전체 배치에 굳은 상수를
  기준으로 삼던 문제를 고쳤습니다. 아티팩트에 있던
  `runaway_light_fraction`을 서빙 경로가 읽지 않아, 짧거나 싼 배치에서
  한 문항이 그 배치 예산의 큰 몫을 가져갈 수 있었습니다.
- `budget_brake` 블록은 선택 필드 `runaway_share`를 받습니다. 필드가
  없으면 이전과 완전히 같게 동작합니다.
