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

## 공개 범위

여기 있는 것은 최종 정책에 이르는 경로입니다. 그 과정에서 접었던 갈래는
코드를 옮기지 않았습니다. 실험 스크립트가 남긴 결정 식별자에는 개발 당시의
내부 일련번호가 섞여 있는데, 이는 리포트 사이의 연결을 유지하기 위한
것입니다.
