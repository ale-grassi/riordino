FROM python:3.14-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        make \
        git \
        tesseract-ocr \
        tesseract-ocr-osd \
        tesseract-ocr-deu \
        tesseract-ocr-fra \
        tesseract-ocr-ita \
        tesseract-ocr-pol \
    && rm -rf /var/lib/apt/lists/*
