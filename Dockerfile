FROM python:3.12

WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    vim \
    curl \
    ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# uv のインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH に uv を追加
ENV PATH="/root/.local/bin:$PATH"

# UV用の環境変数設定
ENV UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy

# プロジェクト全体をコピー
COPY . .

# uv sync で依存関係をインストール
RUN uv sync --frozen

# 仮想環境のPythonパスを設定
ENV PATH="/app/.venv/bin:$PATH"

# srcフォルダを作業ディレクトリとして設定
WORKDIR /app/src
