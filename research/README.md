<!--
SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
SPDX-License-Identifier: Apache-2.0
-->

# 실험 파이프라인

`src/`의 라우터가 읽는 아티팩트를 만들어낸 코드입니다. 제출 이미지에는
들어가지 않으며, 여기서만 numpy를 사용합니다.

```console
python3 -m pip install -r research/requirements.txt
python3 tools/materialize_public_data.py
```

이후 명령은 저장소 루트에서 `PYTHONPATH=.:src:baselines`로 실행합니다.

## 디렉터리

| 경로 | 내용 |
| --- | --- |
| `lab/` | 실험이 공유하는 라이브러리 |
| `experiments/` | 가설 하나에 대응하는 실행 스크립트 |
| `diagnostics/` | 수치가 왜 그런지 확인하는 도구 |
| `audit/` | 규칙 준수와 재현을 확인하는 도구 |
| `export/` | 학습 결과를 런타임 아티팩트로 굳히는 도구 |
| `artifacts/` | 인증 과정이 비교 기준으로 쓰는 참조 아티팩트 |

## 실험 순서

정책을 고르기까지의 흐름입니다. 파일 이름이 그대로 이야기가 됩니다.

```console
python3 -m research.experiments.search_policy_space   # 게이트를 통과하는 후보 탐색
python3 -m research.experiments.rank_expected_score   # 기댓값 기준으로 줄 세우기
python3 -m research.experiments.try_adaptive_guard    # 배치 적응형 가드 시도 (실패)
python3 -m research.experiments.lock_static_caps      # 정적 상한으로 일단 고정
python3 -m research.experiments.try_family_costing    # 계열별 비용 회계 시도
python3 -m research.experiments.select_family_guard   # Train으로 가드 배수 선택
python3 -m research.experiments.lock_final_policy     # Dev 거부권을 걸고 최종 고정
```

`try_adaptive_guard`는 실패한 시도입니다. 지웠다면 다음 사람이 같은 벽에
다시 부딪히므로 남겨 뒀습니다. 배치 구성에 맞춰 예산을 조절하려 했지만
비용 예측 오차가 계열마다 달라 Train에서 배운 보정이 Dev에서 무너졌고,
그 관찰이 `try_family_costing`으로 이어졌습니다.

각 스크립트는 `build/<스크립트 이름>/report.json`에 결과를 씁니다.
`lock_final_policy`가 만든 아티팩트를 런타임에 반영하려면 직접
`src/ossp_router/resources/`로 복사합니다. 실험 코드는 런타임 트리에 쓰지
않습니다.

## 소요 시간

공개 Train 1,760문항과 Dev 880문항 기준입니다. 2코어 x86_64에서 잰
값이므로 절대값보다 상대적인 크기를 참고하십시오.

| 단계 | 대략 |
| --- | ---: |
| 자료 준비 (`materialize_public_data.py`) | 수 분 (네트워크 속도에 좌우) |
| `search_policy_space` | 6분 |
| `rank_expected_score` | 1초 미만 |
| `lock_static_caps` | 4분 |
| `try_family_costing` | 3분 |
| `select_family_guard` | 3분 |
| `lock_final_policy` | 9분 |
| `router_reproduction` | 30초 |
| `submission_contract` | 25초 |

전체 체인은 25분 정도입니다. 후보 하나마다 흔들린 배치 여러 개에서 배분을
다시 돌리는 것이 대부분의 시간을 차지합니다.

`lock_final_policy`가 만든 `build/lock-final-policy/family-guard-router.v1.json`은
`src/ossp_router/resources/family-guard-router.v1.json`과 바이트 단위로
같아야 합니다.

## 검증

```console
python3 -m research.audit.router_reproduction
python3 -m research.audit.submission_contract \
  --router-module ossp_router.family_guard_router \
  --input data/materialized/dev/inputs.json \
  --report build/contract/report.json
```

앞은 실제 라우터가 실험이 고른 정책을 그대로 재현하는지 확인하고, 뒤는
문항 ID와 입력 순서를 바꿔도 선택이 같은지, 등급 사이에 정보가 새지 않는지
확인합니다.

## 결과가 다르게 나온다면

`router_reproduction`이 `ALL_MATCH False`를 내면 순서대로 확인해 보십시오.

1. **자료가 다른 경우.** `data/public-data.v1.json`에 적힌 문항 수와
   SHA-256을 `data/materialized/`의 파일과 비교합니다. 원천 자료를 받는
   과정에서 어긋나면 여기서 드러납니다.
2. **계열 분류가 어긋난 경우.** 출력의 `family mismatch`가 0이 아니면
   런타임과 실험의 프롬프트 분류기가 갈라진 것입니다. 둘은 같은 규칙을
   따라야 합니다.
3. **아티팩트가 다른 경우.** `match_q`는 맞는데 `match_r`만 틀리면 비용
   보정이, 반대면 품질 헤드가 다릅니다.

숫자가 소수점 아래에서만 다르다면 numpy 버전을 확인하십시오. 고정 버전은
`research/requirements.txt`에 있습니다.

## 공개 범위

여기 있는 것은 최종 정책에 이르는 경로입니다. 그 과정에서 접었던 갈래는
코드를 옮기지 않았습니다. 실험 스크립트가 남긴 결정 식별자에는 개발 당시의
내부 일련번호가 섞여 있는데, 이는 리포트 사이의 연결을 유지하기 위한
것입니다.
