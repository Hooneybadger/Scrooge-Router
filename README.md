<!--
SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Scrooge Router

프롬프트만 보고 어떤 언어 모델에 맡길지 정하는 라우터입니다. 쉬운 문항은
싼 모델에 보내고 아낀 예산을 어려운 문항에 몰아주어, 정해진 비용 안에서
정답률을 최대한 끌어올립니다.

이름은 구두쇠 스크루지에서 따왔습니다. 이 라우터의 특징이 비용을 아끼는
방식에 있기 때문입니다. 문항마다 품질과 비용의 분포를 보고, 배치 구성이
흔들리면 사용 비율을 줄입니다. 같은 내용은 항상 같은 모델로 올립니다.

[2026 오픈소스 개발자대회](https://osscontest.kr/) SK텔레콤 지정과제
[Efficient LLM Routing Challenge](https://github.com/sktelecom/ossp-2026-llm-router-challenge)에서
출발했습니다.

## 무엇을 푸는 문제인가

같은 문항이라도 어떤 모델에 보내느냐에 따라 비용과 정답률이 함께 달라집니다.
비싼 모델을 많이 쓰면 정답률은 오르지만 비용이 한도를 넘고, 한도를 넘기면
그 등급은 **0점**이 됩니다. 아껴 쓰면 안전하지만 점수를 놓칩니다.

어려운 지점은 예산 배분 자체가 아니라 **비용 예측이 틀린다는 것**입니다.
비용을 낮게 잡은 문항에 비싼 모델을 배정하면 배치 전체가 한도를 넘습니다.
그래서 상한 하나로 막으려면 예측이 가장 많이 빗나가는 종류를 기준으로
모든 곳에 여유를 잡아야 하고, 결국 예산 대부분을 쓰지 못한 채 남깁니다.

이 라우터는 여유를 문항의 비용 상단과 배치의 구성 편이에 둡니다.

## 결과

등급은 셋이고 각각 비용 한도와 최종 점수 가중치가 다릅니다. 비용 비율은
같은 배치를 전부 가장 싼 모델로 처리했을 때를 1로 둔 상대값입니다.

공개 Dev 880문항 기준입니다.

| 등급 | 비용 비율 | 한도 | 가중치 | 등급 점수 |
| --- | ---: | ---: | ---: | ---: |
| Fast | 1.159 | 1.25 | 0.4 | 0.6787 |
| Balanced | 1.707 | 2.0 | 0.3 | 0.7139 |
| Premium | 3.436 | 4.0 | 0.3 | 0.7670 |

최종 점수 `0.715767`. 세 등급 모두 한도의 95% 아래에서 동작합니다.

실행에는 네트워크도 GPU도 필요하지 않습니다. 표준 라이브러리만 쓰고,
880문항 한 등급을 처리하는 데 2코어에서 2초, 메모리는 42MB입니다.
한도는 등급당 90초와 2GiB입니다.

## 5분 만에 돌려보기

설치할 의존성이 없습니다. Python 3.9 이상이면 됩니다.

```console
git clone https://github.com/Hooneybadger/Scrooge-Router.git
cd Scrooge-Router
```

toy 자료로 세 등급의 선택 결과를 만들고 채점합니다.

```console
for tier in fast balanced premium; do
  PYTHONPATH=src python3 -m ossp_router.distributional_router \
    --input data/toy/inputs.json \
    --tier "$tier" \
    --output "build/toy/$tier.json"
done

PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input data/toy/inputs.json \
  --outcomes data/toy/outcomes.json \
  --submissions build/toy \
  --report build/toy-report.json
```

공개 Dev 880문항으로 위 표의 수치를 재현하려면 먼저 자료를 준비합니다.
일부 원천 자료는 라이선스 조건 때문에 내려받아 결합해야 합니다.

```console
python3 -m venv .venv-data
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
.venv-data/bin/python tools/materialize_public_data.py
```

그다음 `data/toy/inputs.json`을 `data/materialized/dev/inputs.json`으로,
`data/toy/outcomes.json`을 `data/dev/outcomes.json`으로 바꿔 같은 명령을
실행하면 됩니다.

컨테이너로 실행하려면 다음과 같습니다. 공식 평가 플랫폼은 `linux/arm64`입니다.

```console
docker build --pull --platform linux/arm64 \
  --file container/Dockerfile --tag scrooge-router:local .
```

## 어떻게 동작하나

제출 라우터는 `distributional_router` 하나입니다. 학습은 실행 시점에 하지 않고,
동결된 트리와 상한만 읽습니다.

| 단계 | 하는 일 |
| --- | --- |
| 특징 | 길이·문장·기호 같은 구조 특징과 고정 어휘로 문항을 숫자로 바꿈 |
| 예측 | 모델별 품질, 비용 평균, 비용 상단을 트리 앙상블로 예측 |
| 위험 | 배치 구성이 흔들리면 사용 비율을 내리고, Fast는 think 모델을 쓰지 않음 |
| 배분 | 같은 내용은 묶어서 올리고, 이득 대비 비용이 큰 순으로 한도에 닿을 때까지 올림 |

이전 계열 가드·예산 브레이크 계층은 저장소에 비교용으로 남겨 둡니다.
자세한 구조는 [아키텍처](docs/architecture.md)에, 왜 이렇게 정했는지는
[설계 결정 기록](docs/decisions.md)에 있습니다.

## 저장소 구조

| 경로 | 내용 |
| --- | --- |
| `src/ossp_router/` | 제출 이미지에 들어가는 런타임. 표준 라이브러리만 사용 |
| `research/` | 아티팩트를 만든 실험 파이프라인. numpy 사용 |
| `docs/` | 아키텍처, 설계 결정, 로드맵과 대회 규격 문서 |
| `tests/` | 계약 테스트와 저장소 정책 테스트 |
| `container/` | 이미지 빌드 파일 |

## 문서

프로젝트를 이해하려면 다음 순서로 읽으면 됩니다.

- [아키텍처](docs/architecture.md): 제출 라우터와 데이터 흐름
- [설계 결정 기록](docs/decisions.md): 왜 그 값을 골랐는지, 무엇이 실패했는지
- [로드맵과 확장 지점](docs/roadmap.md): 다른 상황에 적용하는 방법
- [실험 파이프라인](research/README.md): 아티팩트를 다시 만드는 절차

대회 규격은 주최측 문서를 그대로 따릅니다.

- [과제 규칙](docs/CHALLENGE_RULES.md)
- [점수 계산](docs/SCORING.md)
- [컨테이너 실행 규격](docs/RUNTIME.md)
- [데이터 카드](docs/DATA_CARD.md)
- [제출 안내](docs/SUBMISSION.md)

## 기여

기여를 환영합니다. 절차와 로컬 검증 명령은
[기여 안내](CONTRIBUTING.md)에 있습니다. 처음이시라면
[`good first issue`](https://github.com/Hooneybadger/Scrooge-Router/labels/good%20first%20issue)
라벨을 보시면 좋습니다.

## 라이선스

코드와 문서는 [Apache License 2.0](LICENSE)으로 제공합니다. 이 라이선스는
제3자 벤치마크 자료를 재라이선스하지 않으며, 자료별 조건은
[DATA_LICENSES.md](DATA_LICENSES.md)에 있습니다.
