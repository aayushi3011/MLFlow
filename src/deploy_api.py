"""
API deployment for Titanic Survival Prediction.
- Loads latest model from MLflow registry (if available)
- Falls back to local model if registry load fails
"""

import mlflow
import mlflow.spark
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pyspark.sql import SparkSession
import warnings

# Suppress Spark/MLflow noisy warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival API")

# Spark session
spark = SparkSession.builder.appName("TitanicAPI").getOrCreate()

# Model name
MODEL_NAME = "TitanicModel"

# Try to load model
model = None
try:
    log.info(f"Trying to load {MODEL_NAME} from MLflow registry...")
    # ⚠️ Stages are deprecated, so we fetch latest version
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        model_uri = latest.source
        model = mlflow.spark.load_model(model_uri)
        log.info(f"✅ Loaded {MODEL_NAME} v{latest.version} from MLflow registry")
    else:
        raise Exception("No versions found in MLflow registry")

except Exception as e:
    log.warning(f"⚠️ Could not load from MLflow registry ({e}). Falling back to local model...")
    model = mlflow.spark.load_model("models/best_rf_model")
    log.info("✅ Local Spark model loaded successfully.")

# Request schema
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    FamilySize: int
    IsSolo: int
    Title: str

@app.post("/predict")
def predict(passenger: Passenger):
    df = pd.DataFrame([passenger.dict()])
    sdf = spark.createDataFrame(df)
    preds = model.transform(sdf).select("prediction").collect()
    return {"prediction": int(preds[0]["prediction"])}

