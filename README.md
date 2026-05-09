# MLOps Batch Job — Rolling-Mean Signal Pipeline

A minimal, reproducible MLOps batch job that ingests OHLCV price data,
computes a rolling-mean signal, and emits structured metrics JSON + a
structured log file. Runs identically on your machine and inside Docker.

---

## Project structure

```
.
├── run.py           # Main pipeline
├── config.yaml      # Job configuration (seed, window, version)
├── data.csv         # 10,000-row OHLCV dataset (BTC/USD, 1-min bars)
├── requirements.txt # Python dependencies
├── Dockerfile       # Container definition
├── metrics.json     # Sample output from a successful local run
├── run.log          # Sample log from a successful local run
└── README.md
```

---

## Local run

### 1. Install dependencies (Python 3.9+)

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

All four flags are required — no paths are hard-coded in the script.

The final metrics JSON is printed to **stdout**; logs go to **run.log**;
the machine-readable metrics file is written to **metrics.json**.

---

## Docker build & run

```bash
# Build
docker build -t mlops-task .

# Run (prints metrics JSON to stdout, exits 0 on success)
docker run --rm mlops-task
```

`data.csv` and `config.yaml` are baked into the image at build time.
The container writes `metrics.json` and `run.log` internally and
streams the final JSON to stdout.

To retrieve the output files from the container:

```bash
docker run --rm -v "$(pwd)/output:/app" mlops-task
# metrics.json and run.log appear in ./output/
```

---

## Configuration (config.yaml)

| Key       | Type    | Description                                 |
|-----------|---------|---------------------------------------------|
| `seed`    | integer | NumPy random seed for reproducibility       |
| `window`  | integer | Rolling-mean window size (rows)             |
| `version` | string  | Pipeline version tag written to metrics JSON|

All three keys are required; the job fails fast with a clear error if any
are missing or have the wrong type.

---

## Signal logic

```
rolling_mean[i] = mean(close[i-window+1 … i])   # NaN for first window-1 rows

signal[i] = 1  if close[i] > rolling_mean[i]
           = 0  otherwise
           = NaN  (warm-up rows, excluded from metrics)
```

The first `window − 1` rows are excluded from `rows_processed` and
`signal_rate` — they are retained in the DataFrame for inspection but
carry `NaN` signals. This is consistent and documented so results are
fully reproducible.

---

## Example metrics.json

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.492,
  "latency_ms": 50,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed = 9996` because window=5 excludes the first 4 warm-up rows
> from a 10,000-row dataset.

### Error output shape

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Columns present: ['timestamp', 'open']"
}
```

Metrics JSON is **always** written — even on failure — so downstream
monitors always have a machine-readable status.

---

## Validation & error handling

The pipeline validates and exits cleanly for:

| Scenario                        | Behaviour                                      |
|---------------------------------|------------------------------------------------|
| Missing input CSV               | `FileNotFoundError` → error metrics JSON       |
| Empty CSV file                  | `ValueError` → error metrics JSON              |
| Invalid CSV format              | `ValueError` with parse details                |
| Missing `close` column          | `ValueError` listing present columns           |
| Non-numeric rows in `close`     | Warning logged, rows dropped, run continues    |
| Missing config file             | `FileNotFoundError` → error metrics JSON       |
| Missing/wrong-type config keys  | `ValueError`/`TypeError` → error metrics JSON  |

---

## Reproducibility

Set `seed` in `config.yaml`. `numpy.random.seed(seed)` is called before
any processing. With identical `seed`, `window`, and input data the
output is bit-for-bit identical across runs and environments.

---

## Exit codes

| Code | Meaning  |
|------|----------|
| `0`  | Success  |
| `1`  | Failure  |
