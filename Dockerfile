FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src ./src
COPY config ./config
COPY README.md LICENSE.txt CONTRIBUTING.MD ./

RUN pip install -e /app/src/practorflow --no-deps

RUN mkdir -p /app/models /app/chroma_db

WORKDIR /app/src
CMD ["python", "sample.py"]
