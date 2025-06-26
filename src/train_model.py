import os
import sys
import glob
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import argparse
import pandas as pd
import pyspark
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
import xgboost as xgb
import mlflow
import mlflow.sklearn

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

    # train-test split
    y_tt = y_df[y_df['snapshot_date'] <= config['train_test_end_date']]
    X_tt = X_df[X_df['customer_id'].isin(y_tt['customer_id'])]

    return X_tt, y_tt, X_oot, y_oot


def train_model(X_tt, y_tt, X_oot, y_oot, config):
    """
    Train XGBoost with optional hyperparameter search and return artefact.
    """
    X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt, 
                                                        test_size=config['train_test_ratio'], 
                                                        random_state=42, 
                                                        shuffle=True, 
                                                        stratify=y_tt['label'])

    X_train = X_train.drop(columns=['customer_id', 'snapshot_date'])
    X_test = X_test.drop(columns=['customer_id', 'snapshot_date'])
    X_oot = X_oot.drop(columns=['customer_id', 'snapshot_date'])

    y_train = y_train['label']
    y_test = y_test['label']
    y_oot = y_oot['label']

    # standard scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_oot_s = scaler.transform(X_oot)

    scorer = make_scorer(roc_auc_score)

    ###### THIS CAN BE A FUNCTION IN ITSELF!!!!! SEPARATE MODELS (LF, RF, AND XGBOOST WITH SPECIFIC HYPERPARAMETERS)
    # define model and param grid
    clf = xgb.XGBClassifier(eval_metric='auc', random_state=88)
    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    # hyperparameter search
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        scoring=scorer,
        n_iter=100,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train_s, y_train)

    # best model
    best = search.best_estimator_
    ###### THIS CAN BE A FUNCTION IN ITSELF!!!!!
    # evaluate
    def auc_score(model, Xd, yd):
        prob = model.predict_proba(Xd)[:, 1]
        return roc_auc_score(yd, prob)

    train_auc = auc_score(best, X_train_s, y_train)
    test_auc = auc_score(best, X_test_s, y_test)
    oot_auc = auc_score(best, X_oot_s, y_oot)

    model_artefact = {}

    model_artefact['model'] = best
    model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['stdscaler'] = scaler
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
    model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc
    model_artefact['results']['auc_test'] = test_auc
    model_artefact['results']['auc_oot'] = oot_auc
    model_artefact['results']['gini_train'] = round(2*train_auc-1,3)
    model_artefact['results']['gini_test'] = round(2*test_auc-1,3)
    model_artefact['results']['gini_oot'] = round(2*oot_auc-1,3)
    model_artefact['hp_params'] = search.best_params_

    return model_artefact


def save_model(artefact, directory='model_bank'):
    """
    Save the model artefact as a pickle file.
    """
    os.makedirs(directory, exist_ok=True)
    version = artefact['model_version']
    path = os.path.join(directory, f"{version}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(artefact, f)
    print(f"Model saved to {path}")

def log_metrics(artefact):
    print("Logging current model metrics...")
    with mlflow.start_run(run_name=artefact['model_version']):
        mlflow.log_metric("auc_train", artefact["results"]["auc_train"])
        mlflow.log_metric("auc_test",  artefact["results"]["auc_test"])
        mlflow.log_metric("auc_oot",   artefact["results"]["auc_oot"])

def main(config_path):
    """
    Main entry point: load config, init Spark, preprocess, train, and save.
    """
    # Get config
    cfg = get_config(config_path)
    # Set up spark
    spark = init_spark()
    X_tt, y_tt, X_oot, y_oot = preprocess_data(cfg, spark)
    artefact = train_model(X_tt, y_tt, X_oot, y_oot, cfg)
    # Set up MLflow
    print("ML Flow is running...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000"))
    mlflow.set_experiment(artefact['model_version'])
    log_metrics(artefact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_train.json", help="Path to config JSON")
    args = parser.parse_args()
    main(args.config)

