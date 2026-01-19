FROM python:3.12

WORKDIR /app

# 必要なパッケージと日本語フォントをインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    vim \
    curl \
    ca-certificates \
    fonts-noto-cjk \
    fonts-takao-gothic \
    fontconfig \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && fc-cache -fv

# uv のインストール（pipを使用）
RUN pip install --no-cache-dir uv && \
    uv --version

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
