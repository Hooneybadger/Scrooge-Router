# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11.15-alpine3.23@sha256:f73754c398b259dfbbe482361dca8b464dea57da74efe5214966ca2ee767ee12 AS fnv-builder

RUN apk add --no-cache build-base
COPY container/fnvfast.c /build/fnvfast.c
RUN cc -O3 -shared -fPIC $(python3-config --includes) \
    /build/fnvfast.c -lm -o /build/_fnvfast.so

FROM python:3.11.15-alpine3.23@sha256:f73754c398b259dfbbe482361dca8b464dea57da74efe5214966ca2ee767ee12

# 실행 시 패키지를 설치하지 않으므로 빌드 전용 도구를 최종 이미지에서 제거합니다.
RUN python3 -m pip uninstall --yes pip setuptools wheel

ARG SOURCE_MANIFEST_SHA256=unbound

LABEL io.sktelecom.ossp.source-manifest-sha256="${SOURCE_MANIFEST_SHA256}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/router \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp

COPY --chown=65532:65532 src container/entrypoint.py /opt/router/
COPY --from=fnv-builder --chown=65532:65532 \
    /build/_fnvfast.so /opt/router/ossp_router/_fnvfast.so

WORKDIR /opt/router

USER 65532:65532

ENTRYPOINT ["python3", "/opt/router/entrypoint.py"]
