# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.9-slim

# Keeps Python from buffering stdout/stderr (important for live log visibility)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source + data ─────────────────────────────────────────────────────────────
COPY run.py       .
COPY config.yaml  .
COPY data.csv     .

# ── Run ───────────────────────────────────────────────────────────────────────
# Exit code is propagated from run.py (0 = success, 1 = error)
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
