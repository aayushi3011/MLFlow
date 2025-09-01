"""
Titanic data preprocessing with PySpark.
- Loads raw CSV files
- Performs feature engineering
- Handles missing values
- Removes unused columns
- Saves cleaned train/test splits
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, regexp_extract, lit
from pyspark.sql.types import DoubleType, IntegerType

# Start Spark session
spark = SparkSession.builder.appName("TitanicDataPreprocessing").getOrCreate()

# === Load raw data ===
train_df = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)
test_df = spark.read.csv("data/raw/test.csv", header=True, inferSchema=True)

# Add dataset indicator before combining
train_df = train_df.withColumn("dataset", lit("train"))
test_df = test_df.withColumn("dataset", lit("test"))
all_data = train_df.unionByName(test_df, allowMissingColumns=True)

# === Feature engineering ===
all_data = all_data.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
all_data = all_data.withColumn("IsSolo", when(col("FamilySize") == 1, 1).otherwise(0))
all_data = all_data.withColumn("Title", regexp_extract(col("Name"), r' ([A-Za-z]+)\.', 1))

# === Missing values ===
avg_age = all_data.select(mean(col("Age"))).collect()[0][0]
avg_fare = all_data.select(mean(col("Fare"))).collect()[0][0]

all_data = all_data.na.fill({
    "Age": avg_age,
    "Embarked": "S",
    "Fare": avg_fare
})

# Fill missing Cabin with "Unknown"
all_data = all_data.withColumn("Cabin", when(col("Cabin").isNull(), "Unknown").otherwise(col("Cabin")))

# === Drop unused columns ===
all_data = all_data.drop("Name", "Ticket", "Cabin")

# === Split back into train/test ===
train_final = all_data.filter(col("dataset") == "train").drop("dataset")
test_final = all_data.filter(col("dataset") == "test").drop("dataset", "Survived")

# === Save processed outputs ===
train_final.write.csv("data/processed/train_processed.csv", header=True, mode="overwrite")
test_final.write.csv("data/processed/test_processed.csv", header=True, mode="overwrite")

print("Preprocessing completed. Cleaned data saved to data/processed/")

# Stop Spark session
spark.stop()
