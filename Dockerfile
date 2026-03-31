FROM python:3.14-slim

ARG LANGS=""

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        make \
        tesseract-ocr \
        tesseract-ocr-osd \
        $(for l in $LANGS; do echo "tesseract-ocr-$l"; done) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Makefile pyproject.toml README.md LICENSE ./
COPY riordino.py ./
COPY prompts/ ./prompts/

RUN make install PYTHON=python3

ENTRYPOINT ["riordino"]
