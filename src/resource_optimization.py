"""
Resource optimization script for Titanic training.
- Runs training with different Spark + RandomForest settings
- Logs runtime and AUC
- Saves results to reports/resource_optimization_results.csv
"""

import os
import time
import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Logging setup
os.makedirs("reports", exist_ok=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ResourceOptimization")

# Experiment settings
spark_configs = [
    {"executor_memory": "1g", "driver_memory": "1g"},
    {"executor_memory": "2g", "driver_memory": "2g"},
]
rf_configs = [
    {"numTrees": 10, "maxDepth": 5},
    {"numTrees": 20, "maxDepth": 10},
]

results = []

# Loop through configs
for scfg in spark_configs:
    for rfcfg in rf_configs:
        log.info(f"=== Running with Spark {scfg}, RF {rfcfg} ===")

        # Start Spark with given config
        spark = SparkSession.builder \
            .appName("ResourceOptimization") \
            .config("spark.executor.memory", scfg["executor_memory"]) \
            .config("spark.driver.memory", scfg["driver_memory"]) \
            .getOrCreate()

        try:
            # Load data
            df = spark.read.csv("data/processed/train_processed.csv", header=True, inferSchema=True)

            # Features
            categorical_cols = ["Sex", "Embarked", "Title"]
            numerical_cols = ["Age", "Fare", "Pclass", "FamilySize", "IsSolo"]

            stages = []
            for col_name in categorical_cols:
                indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx", handleInvalid="keep")
                encoder = OneHotEncoder(inputCol=f"{col_name}_idx", outputCol=f"{col_name}_vec")
                stages += [indexer, encoder]

            assembler = VectorAssembler(
                inputCols=numerical_cols + [f"{c}_vec" for c in categorical_cols],
                outputCol="features"
            )
            rf = RandomForestClassifier(
                labelCol="Survived", featuresCol="features",
                numTrees=rfcfg["numTrees"], maxDepth=rfcfg["maxDepth"]
            )
            stages += [assembler, rf]

            pipeline = Pipeline(stages=stages)

            # Train/test split
            train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

            evaluator = BinaryClassificationEvaluator(labelCol="Survived")

            # Run experiment
            start = time.time()
            model = pipeline.fit(train_df)
            preds = model.transform(val_df)
            auc = evaluator.evaluate(preds)
            duration = time.time() - start

            log.info(f"AUC={auc:.4f}, Time={duration:.2f}s")

            results.append({
                "executor_memory": scfg["executor_memory"],
                "driver_memory": scfg["driver_memory"],
                "numTrees": rfcfg["numTrees"],
                "maxDepth": rfcfg["maxDepth"],
                "AUC": auc,
                "Runtime_sec": duration
            })

        except Exception as e:
            log.error(f"Experiment failed: {e}")
        finally:
            spark.stop()

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("reports/resource_optimization_results.csv", index=False)
log.info("✅ Resource optimization completed. Results saved to reports/resource_optimization_results.csv")
