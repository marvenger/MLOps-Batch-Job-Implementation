"""
MLOps Batch Job — Rolling-Mean Signal Pipeline
Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling-mean binary signal pipeline"
    )
    parser.add_argument("--input",    required=True, help="Path to input CSV")
    parser.add_argument("--config",   required=True, help="Path to YAML config")
    parser.add_argument("--output",   required=True, help="Path for metrics JSON output")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for log file output")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler — DEBUG and above
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open() as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    if not isinstance(cfg["seed"], int):
        raise TypeError(f"Config 'seed' must be an integer, got {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got {cfg['window']!r}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"Config 'version' must be a non-empty string, got {cfg['version']!r}")

    logger.info(
        "Config loaded — version=%s  seed=%d  window=%d",
        cfg["version"], cfg["seed"], cfg["window"]
    )
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV '{input_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"CSV parsed but contains no rows: {input_path}")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Columns present: {list(df.columns)}"
        )

    # Coerce close to numeric; flag non-parseable values
    original_len = len(df)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    bad_rows = df["close"].isna().sum()
    if bad_rows:
        logger.warning(
            "%d row(s) with non-numeric 'close' values will be dropped", bad_rows
        )
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    logger.info("Dataset loaded — %d rows (dropped %d invalid)", len(df), original_len - len(df))
    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute rolling mean of 'close' over `window` periods.
    The first (window-1) rows produce NaN — they are retained in the
    DataFrame but excluded from signal computation and metrics.
    """
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()
    nan_count = df["rolling_mean"].isna().sum()
    logger.info(
        "Rolling mean computed — window=%d  warm-up rows excluded=%d",
        window, nan_count
    )
    return df


def compute_signal(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    signal = 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN are assigned signal = NaN (excluded
    from the signal_rate metric).
    """
    df = df.copy()
    mask = df["rolling_mean"].notna()
    df.loc[mask, "signal"] = (df.loc[mask, "close"] > df.loc[mask, "rolling_mean"]).astype(int)
    df.loc[~mask, "signal"] = np.nan

    valid = df["signal"].notna().sum()
    ones  = df["signal"].sum()
    logger.info(
        "Signal generated — valid rows=%d  signal=1 count=%d  signal=0 count=%d",
        valid, int(ones), int(valid - ones)
    )
    return df


# ---------------------------------------------------------------------------
# Metrics output
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict, logger: logging.Logger) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Metrics written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    start_ts = time.time()
    logger.info("=" * 60)
    logger.info("Job starting")
    logger.info("  input   : %s", args.input)
    logger.info("  config  : %s", args.config)
    logger.info("  output  : %s", args.output)
    logger.info("  log-file: %s", args.log_file)
    logger.info("=" * 60)

    # We always want a version in error output; default before config load
    version = "unknown"

    try:
        # ── 1. Config ────────────────────────────────────────────────────
        cfg     = load_config(args.config, logger)
        version = cfg["version"]
        seed    = cfg["seed"]
        window  = cfg["window"]

        # ── 2. Reproducibility seed ──────────────────────────────────────
        np.random.seed(seed)
        logger.debug("NumPy random seed set to %d", seed)

        # ── 3. Dataset ───────────────────────────────────────────────────
        df = load_dataset(args.input, logger)

        # ── 4. Rolling mean ──────────────────────────────────────────────
        df = compute_rolling_mean(df, window, logger)

        # ── 5. Signal ────────────────────────────────────────────────────
        df = compute_signal(df, logger)

        # ── 6. Metrics ───────────────────────────────────────────────────
        valid_mask    = df["signal"].notna()
        rows_processed = int(valid_mask.sum())
        signal_rate   = round(float(df.loc[valid_mask, "signal"].mean()), 4)
        latency_ms    = int((time.time() - start_ts) * 1000)

        logger.info(
            "Metrics — rows_processed=%d  signal_rate=%.4f  latency_ms=%d",
            rows_processed, signal_rate, latency_ms
        )

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          signal_rate,
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        write_metrics(args.output, metrics, logger)
        logger.info("Job completed successfully")
        logger.info("=" * 60)

        # Print final JSON to stdout (Docker requirement)
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:
        logger.error("Job failed: %s", exc, exc_info=True)

        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
        }
        try:
            write_metrics(args.output, error_metrics, logger)
        except Exception as write_exc:
            logger.error("Additionally failed to write error metrics: %s", write_exc)

        logger.info("=" * 60)
        print(json.dumps(error_metrics, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
