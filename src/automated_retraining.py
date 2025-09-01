"""
Automated retraining script for Titanic model.
- Re-runs preprocessing
- Retrains model using Spark MLlib
- Logs and registers new version in MLflow
- Saves retraining logs to reports/retrain_log.txt
Triggered by drift detection or manual execution.
"""

import os
import logging

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# Configure logging (console + file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("reports/retrain_log.txt", mode="a")  # append to file
    ]
)
log = logging.getLogger("AutoRetraining")


def run_step(command: str):
    """Helper to run a shell command with logging."""
    log.info(f"➡️ Running: {command}")
    status = os.system(command)
    if status != 0:
        log.error(f"❌ Step failed: {command}")
        raise RuntimeError(f"Command failed: {command}")
    else:
        log.info(f"✅ Step completed: {command}")


if __name__ == "__main__":
    try:
        log.info("=== 🚀 Automated Retraining Pipeline Started ===")

        # Step 1: Re-run preprocessing
        run_step("python src/preprocess_data.py")

        # Step 2: Retrain model (logs new version in MLflow)
        run_step("python src/train_model.py")

        log.info("🎯 Retraining pipeline finished successfully!")

    except Exception as e:
        log.error(f"Retraining failed: {e}")
