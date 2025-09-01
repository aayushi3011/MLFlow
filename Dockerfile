# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Java (needed for Spark)
RUN apt-get update && apt-get install -y openjdk-11-jdk curl && rm -rf /var/lib/apt/lists/*

# Set Java environment variables for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY dvc.yaml .
COPY README.md .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "src.deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
