#!/usr/bin/env python3
"""
MLOps Task 0: Minimal batch job for signal generation from OHLCV data.
Features: reproducibility, observability, deployment readiness.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to both file and console."""
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")
    
    # Check if config is None or empty
    if config is None:
        raise ValueError("Config file is empty")
    
    # Validate required fields
    required_fields = ['seed', 'window', 'version']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")
    
    # Validate types
    if not isinstance(config['seed'], int):
        raise ValueError(f"seed must be an integer, got {type(config['seed'])}")
    if not isinstance(config['window'], int) or config['window'] < 1:
        raise ValueError(f"window must be a positive integer, got {config['window']}")
    if not isinstance(config['version'], str):
        raise ValueError(f"version must be a string, got {type(config['version'])}")
    
    # Set random seed for reproducibility
    np.random.seed(config['seed'])
    
    logger.info(f"Config loaded and validated: seed={config['seed']}, "
                f"window={config['window']}, version={config['version']}")
    
    return config


def load_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load and validate the input CSV file."""
    logger.info(f"Loading data from {input_path}")
    
    # Check file exists
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check file is not empty
    if input_file.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file contains no data: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format in {input_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {input_path}: {e}")
    
    # Check if dataframe is empty
    if len(df) == 0:
        raise ValueError("Dataset contains no rows")
    
    # Validate required columns
    if 'close' not in df.columns:
        raise ValueError(f"Required column 'close' not found in dataset. "
                       f"Available columns: {list(df.columns)}")
    
    # Validate close column has valid data
    if df['close'].isna().all():
        raise ValueError("Column 'close' contains only NaN values")
    
    rows_loaded = len(df)
    logger.info(f"Data loaded successfully: {rows_loaded} rows, {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")
    logger.debug(f"First few rows:\n{df.head()}")
    logger.debug(f"Close stats: min={df['close'].min():.2f}, max={df['close'].max():.2f}, "
                 f"mean={df['close'].mean():.2f}")
    
    return df


def compute_signal(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """Compute rolling mean and generate binary signals."""
    logger.info(f"Computing rolling mean on 'close' with window={window}")
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Compute rolling mean - first window-1 rows will be NaN
    result_df['rolling_mean'] = result_df['close'].rolling(window=window, min_periods=window).mean()
    
    # Generate signal: 1 if close > rolling_mean, else 0
    # For NaN rolling_mean values, signal will be NaN (excluded from metrics)
    result_df['signal'] = np.where(
        result_df['rolling_mean'].notna(),
        (result_df['close'] > result_df['rolling_mean']).astype(int),
        np.nan
    )
    
    valid_signals = result_df['signal'].notna().sum()
    nan_signals = result_df['signal'].isna().sum()
    
    logger.info(f"Rolling mean computed: {valid_signals} valid rows, "
                f"{nan_signals} rows with NaN (first {window-1} rows)")
    logger.debug(f"Signal distribution: {(result_df['signal'].value_counts().to_dict())}")
    
    return result_df


def calculate_metrics(df: pd.DataFrame, config: Dict[str, Any], 
                     start_time: float, logger: logging.Logger) -> Dict[str, Any]:
    """Calculate and structure output metrics."""
    logger.info("Calculating metrics")
    
    # Calculate latency in milliseconds
    latency_ms = round((time.time() - start_time) * 1000)
    
    # Calculate signal rate (excluding NaN values)
    valid_signals = df['signal'].dropna()
    
    if len(valid_signals) > 0:
        signal_rate = round(float(valid_signals.mean()), 4)
    else:
        signal_rate = 0.0
        logger.warning("No valid signals to calculate rate")
    
    metrics = {
        "version": config['version'],
        "rows_processed": int(len(df)),
        "metric": "signal_rate",
        "value": signal_rate,
        "latency_ms": latency_ms,
        "seed": int(config['seed']),
        "status": "success"
    }
    
    logger.info(f"Metrics summary: rows_processed={metrics['rows_processed']}, "
                f"signal_rate={metrics['value']}, latency_ms={metrics['latency_ms']}")
    
    return metrics


def write_metrics(metrics: Dict[str, Any], output_path: str, logger: logging.Logger):
    """Write metrics to JSON file and print to stdout."""
    logger.info(f"Writing metrics to {output_path}")
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print to stdout for Docker logs
    print(json.dumps(metrics, indent=2))
    logger.debug(f"Metrics written successfully: {metrics}")


def write_error_metrics(version: str, error_message: str, output_path: str, 
                       logger: logging.Logger):
    """Write error state metrics to JSON file."""
    error_metrics = {
        "version": version,
        "status": "error",
        "error_message": error_message
    }
    
    logger.error(f"Writing error metrics: {error_message}")
    
    try:
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(error_metrics, f, indent=2)
        
        # Print to stdout
        print(json.dumps(error_metrics, indent=2))
    except Exception as e:
        logger.critical(f"Failed to write error metrics: {e}")
        # Last resort - try to print to stdout
        print(json.dumps({"status": "error", "error_message": f"Critical failure: {e}"}))


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='MLOps Task 0: Generate trading signals from OHLCV data'
    )
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--output', required=True, help='Path for output metrics JSON')
    parser.add_argument('--log-file', required=True, help='Path for log file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("MLOps Task 0 - Signal Generation Pipeline Started")
    logger.info("=" * 60)
    logger.info(f"Configuration: input={args.input}, config={args.config}, "
                f"output={args.output}, log_file={args.log_file}")
    
    config = None
    version = "v1"  # Default fallback version
    
    try:
        # Step 1: Load and validate config
        config = load_config(args.config, logger)
        version = config['version']
        
        # Step 2: Load and validate dataset
        logger.info("Step 1/4: Loading dataset")
        df = load_data(args.input, logger)
        
        # Step 3: Compute rolling mean and signal
        logger.info("Step 2/4: Computing rolling mean")
        df_processed = compute_signal(df, config['window'], logger)
        
        # Step 4: Generate binary signal
        logger.info("Step 3/4: Generating signals")
        # Signal generation already done in compute_signal
        
        # Step 5: Calculate metrics
        logger.info("Step 4/4: Computing metrics")
        metrics = calculate_metrics(df_processed, config, start_time, logger)
        
        # Step 6: Write output
        write_metrics(metrics, args.output, logger)
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully")
        logger.info(f"Total processing time: {metrics['latency_ms']}ms")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        error_message = f"File not found: {str(e)}"
        logger.error(error_message, exc_info=True)
        write_error_metrics(version, error_message, args.output, logger)
        return 1
        
    except ValueError as e:
        error_message = f"Validation error: {str(e)}"
        logger.error(error_message, exc_info=True)
        write_error_metrics(version, error_message, args.output, logger)
        return 1
        
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        logger.error(error_message, exc_info=True)
        write_error_metrics(version, error_message, args.output, logger)
        return 1


if __name__ == "__main__":
    sys.exit(main())
