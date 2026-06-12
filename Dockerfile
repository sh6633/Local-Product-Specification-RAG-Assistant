FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_EXTRA_INDEX_URL="https://abetlen.github.io/llama-cpp-python/whl/cu124"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml uv.lock ./
RUN uv python install 3.11
RUN uv sync --frozen --python 3.11
RUN uv pip install --no-cache-dir --force-reinstall llama-cpp-python==0.3.23

COPY src ./src
COPY data ./data
COPY README.md ./

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
