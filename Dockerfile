FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1

FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  python3-venv \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /app/.venv

ENV UV_LINK_MODE=copy
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install -r requirements.txt \
  --index-strategy unsafe-best-match \
  --no-deps

COPY . .
RUN uv pip install .[mineru,paddlepaddle]

FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  ccache \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

RUN mkdir -p /root/.paddleocr
VOLUME [ "/root/.paddleocr" ]

COPY entrypoint.sh /app/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["bash", "/app/entrypoint.sh"]

CMD ["--host=0.0.0.0", "--port=8000"]
