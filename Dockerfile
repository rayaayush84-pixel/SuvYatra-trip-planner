FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY manage.py .

RUN useradd -m appuser && mkdir -p /app/chroma_db && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# One worker: ChromaDB's local persistent store is not built for many writer processes.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
