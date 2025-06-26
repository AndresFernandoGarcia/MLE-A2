import os
import sys
import glob
import pandas as pd
import pickle
import numpy as np
import json
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_config(config_path):   
    """
    Load and return the configuration dictionary.
    """ 
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
                # parse dates
        cfg['model_train_date'] = datetime.strptime(cfg['model_train_date_str'], "%Y-%m-%d")
        cfg['oot_end_date'] = cfg['model_train_date'] - timedelta(days=1)
        cfg['oot_start_date'] = cfg['model_train_date'] - relativedelta(months=cfg['oot_period_months'])
        cfg['train_test_end_date'] = cfg['oot_start_date'] - timedelta(days=1)
        cfg['train_test_start_date'] = cfg['oot_start_date'] - relativedelta(months=cfg['train_test_period_months'])
        print("Config file loaded succesfully!")
        return cfg

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)

def init_spark(app_name="dev"):
    """
    Initialize and return a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def preprocess_data(config, spark):
    """
    Load, filter, and split data into train-test and OOT sets.
    Returns X_train_test, y_train_test, X_oot, y_oot as pandas DataFrames/Series.
    """
    # load label store
    folder_path = "datamart/gold/label_store/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    label_store_sdf = spark.read.option("header", "true").parquet(*files_list)

    # filter by date window
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))
    
    feature_location = "datamart/gold/feature_store"
    features_files_list = [
        os.path.join(feature_location, os.path.basename(f))
        for f in glob.glob(os.path.join(feature_location, '*'))
    ]

    # Load CSV into DataFrame - connect to feature store
    features_store_sdf = spark.read.option("header", "true").parquet(*features_files_list)

    # extract feature store
    features_sdf = features_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))

    y_df = labels_sdf.toPandas().sort_values(by='customer_id')
    X_df = features_sdf.toPandas().sort_values(by='customer_id')

    X_df['snapshot_date'] = pd.to_datetime(X_df['snapshot_date'])
    y_df['snapshot_date'] = pd.to_datetime(y_df['snapshot_date'])

    # Simulate inner merge to remove non-existent customer_id's in x and y to standardize df size
    X_df = X_df[np.isin(X_df['customer_id'], y_df['customer_id'].unique())]
    y_df = y_df[np.isin(y_df['customer_id'], X_df['customer_id'].unique())]

    # OOT split
    y_oot = y_df[
        (y_df['snapshot_date'] >= config['oot_start_date']) &
        (y_df['snapshot_date'] <= config['oot_end_date'])
    ]
    X_oot = X_df[X_df['customer_id'].isin(y_oot['customer_id'])]

    return X_oot, y_oot

def load_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model_artefact = pickle.load(file)
    print("Model Loaded Successfully!\nCurrent Model:", model_path)
    return loaded_model_artefact

def eval_model(loaded_model_artefact, X_oot, y_oot):
    # Drop ID columns if present
    X_oot_clean = X_oot.drop(['customer_id', 'snapshot_date'], axis=1, errors='ignore')

    # Apply stored scaler
    scaler = loaded_model_artefact['preprocessing_transformers']['stdscaler']
    X_oot_processed = scaler.transform(X_oot_clean)

    # Predict
    y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot_processed)[:, 1]
    oot_auc_score = roc_auc_score(y_oot['label'], y_pred_proba)
    print("OOT AUC score: ", oot_auc_score)


def main(config_path):
    """
    Main entry point: load config, init Spark, preprocess, train, and save.
    """
    spark = init_spark()
    cfg = get_config(config_path)
    version = 'credit_model_' + cfg['model_train_date_str'].replace('-', '_')
    model_path = os.path.join("model_bank/", f"{version}.pkl")
    model = load_model(model_path)
    X_oot, y_oot = preprocess_data(cfg, spark)
    eval_model(model, X_oot, y_oot)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_train.json", help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)