# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependency list and install (offline-friendly: all wheels are manylinux)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY pdf_outline_extractor.py /app/pdf_outline_extractor.py
# (Optional) copy your model if you have one
# COPY stage1_final_model.joblib /app/stage1_final_model.joblib

# Default: batch mode on /app/input -> /app/output
ENTRYPOINT ["python", "/app/pdf_outline_extractor.py"]
