# MLOps-Batch-Job-Implementation

# MLOps Task 0 - Trading Signal Pipeline

A minimal MLOps-style batch job demonstrating reproducibility, observability, and deployment readiness for trading signal generation.

## 📋 Overview

This pipeline processes OHLCV (Open, High, Low, Close, Volume) data to:
1. Load and validate configuration from YAML
2. Read 10,000 rows of Bitcoin price data
3. Compute rolling mean on close prices
4. Generate binary trading signals (1 when close > rolling mean, 0 otherwise)
5. Output structured metrics and detailed logs
6. Run deterministically using configurable random seed

## 🚀 Quick Start

### Local Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log

# View results
cat metrics.json
