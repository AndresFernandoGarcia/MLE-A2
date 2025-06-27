import os
import sys
import glob
import json
import pandas as pd
import pickle
from datetime import datetime
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_config(config_path):
    """
    Load inference configuration.
    Expects JSON with keys: model_path, start_date_str, end_date_str, optional output_path.
    """
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to load inference config: {e}")
        sys.exit(1)

    # Parse dates
    cfg['start_date'] = datetime.strptime(cfg['start_date_str'], "%Y-%m-%d")
    cfg['end_date'] = datetime.strptime(cfg['end_date_str'], "%Y-%m-%d")
    return cfg


def init_spark(app_name="inference"):
    """
    Initialize and return a SparkSession for inference.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def preprocess_inference_data(cfg, spark):
    """
    Load feature data between start_date and end_date.
    Returns a pandas DataFrame with customer_id, snapshot_date, and features.
    """
    feature_location = "datamart/gold/feature_store"
    files = glob.glob(os.path.join(feature_location, '*'))
    files_list = [os.path.join(feature_location, os.path.basename(f)) for f in files]

    sdf = spark.read.option("header", "true").parquet(*files_list)
    filtered = sdf.filter(
        (col("snapshot_date") >= cfg['start_date']) &
        (col("snapshot_date") <= cfg['end_date'])
    )

    pdf = filtered.toPandas()
    pdf['snapshot_date'] = pd.to_datetime(pdf['snapshot_date'])
    pdf = pdf.sort_values(by='customer_id')
    return pdf


def load_model(model_path):
    """
    Load the pkl model artefact.
    """
    try:
        with open(model_path, 'rb') as file:
            loaded = pickle.load(file)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        sys.exit(1)

    print(f"Model loaded successfully from: {model_path}")
    return loaded


def predict_batch(model_artefact, X_df):
    """
    Apply preprocessing and predict binary labels for each customer.
    Returns a DataFrame with columns: customer_id, prediction.
    """
    # Drop non-feature columns
    X_clean = X_df.drop(['customer_id', 'snapshot_date'], axis=1, errors='ignore')

    # Apply stored scaler
    scaler = model_artefact['preprocessing_transformers']['stdscaler']
    X_proc = scaler.transform(X_clean)

    # Predict probabilities
    proba = model_artefact['model'].predict_proba(X_proc)[:, 1]
    labels = (proba >= 0.5).astype(int)

    return pd.DataFrame({
        'customer_id': X_df['customer_id'],
        'prediction': labels
    })


def save_predictions(df, cfg):
    """
    Save predictions to CSV. Uses output_path from config or defaults.
    """
    output_path = cfg.get(
        'output_path',
        f"predictions_{cfg['start_date_str']}_{cfg['end_date_str']}.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"Predictions written to: {output_path}")


def main(config_path):
    spark = init_spark()
    cfg = get_config(config_path)

    model_artefact = load_model(cfg['model_path'])
    X_df = preprocess_inference_data(cfg, spark)
    preds_df = predict_batch(model_artefact, X_df)
    save_predictions(preds_df, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for credit model.")
    parser.add_argument("--config", type=str, default="configs/default_inference.json", help="Path to inference config JSON")
    args = parser.parse_args()
    main(args.config)
