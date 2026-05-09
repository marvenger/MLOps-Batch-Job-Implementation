# MLOps-Batch-Job-Implementation

# MLOps Task 0 - Signal Generation Pipeline

Minimal MLOps-style batch job demonstrating reproducibility, observability, and deployment readiness.

## Overview

This pipeline:
1. Loads OHLCV data from CSV
2. Computes rolling mean on close prices
3. Generates binary signals (1 if close > rolling_mean, else 0)
4. Outputs structured metrics and detailed logs
5. Runs deterministically using configurable seed

## Requirements

- Python 3.9+
- Docker (for containerized execution)

## Local Run Instructions

### Setup
```bash
pip install -r requirements.txt
