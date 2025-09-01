"""
Drift detection script for Titanic dataset.
- Compares feature distributions between training and new data using KS-test
- If drift is detected, triggers automated retraining
- Logs events into reports/retrain_log.txt
"""

import pandas as pd
import logging
import os
from scipy.stats import ks_2samp
import subprocess

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# Configure logging (console + file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("reports/retrain_log.txt", mode="a")  # append to log file
    ]
)
log = logging.getLogger("DriftDetection")

# Paths to processed datasets
train_path = "data/processed/train_processed.csv"
new_path = "data/processed/test_processed.csv"

# Helper: load Spark CSV outputs (may have multiple part-*.csv files)
def load_spark_csv(folder_path):
    return pd.concat(
        [pd.read_csv(os.path.join(folder_path, f))
         for f in os.listdir(folder_path) if f.endswith(".csv")],
        ignore_index=True
    )

# Load datasets
train_df = load_spark_csv(train_path)
new_df = load_spark_csv(new_path)

# Features to check drift
features_to_check = ["Age", "Fare", "Pclass"]

for feature in features_to_check:
    stat, p_value = ks_2samp(train_df[feature], new_df[feature])
    log.info(f"KS-test for {feature}: statistic={stat:.4f}, p_value={p_value:.4f}")

    if p_value < 0.05:
        msg = f"⚠️ Drift detected in {feature}! Triggering retraining..."
        log.warning(msg)

        # Append to retrain_log.txt explicitly
        with open("reports/retrain_log.txt", "a") as f:
            f.write(f"{msg}\n")

        # Call automated retraining
        subprocess.call(["python", "src/automated_retraining.py"])
    else:
        log.info(f"No significant drift in {feature}.")
