<!--
SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
SPDX-License-Identifier: Apache-2.0
-->

# 기여 안내

기여를 환영합니다. 버그 신고, 문서 개선, 새 아이디어 제안 모두 좋습니다.

## 시작하기

작업을 시작하기 전에 이슈를 먼저 열어주세요. 이미 누군가 같은 것을 보고
있거나, 더 간단한 방법이 있을 수 있습니다. 오타 수정이나 문서 문장 다듬기는
바로 PR을 보내주셔도 됩니다.

처음이시라면 `good first issue` 라벨이 붙은 이슈부터 보시면 좋습니다.

## 개발 환경

Python 3.9 이상이면 됩니다. 라우터는 표준 라이브러리만 사용하므로 설치할
의존성이 없습니다.

```console
git clone https://github.com/Hooneybadger/Scrooge-Router.git
cd Scrooge-Router
PYTHONPATH=src python3 -m unittest discover -s tests
```

## 보내기 전에 확인할 것

PR을 열면 CI가 같은 검사를 다시 실행합니다. 로컬에서 먼저 확인해주세요.

```console
umask 0022 && PYTHONPATH=src:baselines python3 -m unittest discover -s tests
ruff check .
reuse lint
```

`umask 0022`가 필요한 이유는 런타임 검증이 group과 other에 쓰기 권한이 열린
경로를 거부하기 때문입니다.

`ruff`와 `reuse`가 없다면 다음으로 설치할 수 있습니다.

```console
python3 -m pip install ruff reuse
```

## 라우터 코드를 바꾸는 경우

`src/`의 코드는 대회 평가 이미지에 그대로 들어갑니다. 다음 제약을 지켜주세요.

- 실행 중 네트워크에 접근하거나 파일을 내려받지 않습니다.
- 프롬프트 내용만 사용합니다. 문항 ID, 입력 순서, 정답, 평가 결과는 읽지
  않습니다. 같은 프롬프트는 순서가 바뀌어도 같은 모델을 골라야 합니다.
- 세 등급의 비용 한도를 넘으면 해당 등급은 0점이 됩니다. 비용에 영향을 주는
  변경은 세 등급의 실현 비용 비율을 함께 보고해주세요.

## 실험을 제안하는 경우

이 프로젝트는 실험을 돌리기 전에 통과 기준을 먼저 정합니다. 결과를 본 뒤에
기준을 정하면 공개 Dev 점수에 맞춰진 결론이 나오기 때문입니다. 이슈에
가설과 함께 어떤 수치면 채택하고 어떤 수치면 접을지 적어주세요.

## 커밋과 PR

커밋 메시지는 무엇을 왜 바꿨는지 한 줄로 적어주세요. 형식을 강제하지는
않습니다.

PR 제목은 변경 내용이 드러나게 적고, 관련 이슈가 있으면 본문에
`Closes #번호`로 연결해주세요.

## 라이선스

기여한 내용은 Apache-2.0으로 배포됩니다. 새로 만드는 파일에는 형식에 맞는
SPDX 정보를 넣어주세요.

```text
SPDX-FileCopyrightText: Copyright 2026 Scrooge Router contributors
SPDX-License-Identifier: Apache-2.0
```

데이터 파일을 추가할 때는 [`DATA_LICENSES.md`](DATA_LICENSES.md)의 조건을
확인해주세요. 저장소의 Apache-2.0 라이선스가 데이터셋까지 덮지는 않습니다.
