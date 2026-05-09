FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Create output directory
RUN mkdir -p /app/output

# Set the entrypoint
ENTRYPOINT ["python", "run.py", \
    "--input", "data.csv", \
    "--config", "config.yaml", \
    "--output", "/app/output/metrics.json", \
    "--log-file", "/app/output/run.log"]

# Default command (can be overridden)
CMD []
