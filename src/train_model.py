import os
import sys
import glob
import pickle
import json
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_config(config_path):
    """
    Load training configuration. Expects keys:
      start_date, oot_period_months, train_test_period_months,
      train_test_ratio, model_type, search_space.
    """
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)

    # parse dates
    cfg['start_date_str'] = cfg['start_date']
    cfg['start_date'] = datetime.strptime(cfg['start_date'], "%Y-%m-%d")
    cfg['oot_end_date'] = cfg['start_date'] - timedelta(days=1)
    cfg['oot_start_date'] = cfg['start_date'] - relativedelta(months=cfg['oot_period_months'])
    cfg['train_test_end_date'] = cfg['oot_start_date'] - timedelta(days=1)
    cfg['train_test_start_date'] = cfg['oot_start_date'] - relativedelta(months=cfg['train_test_period_months'])

    # validate model_type
    valid_models = ['xgboost', 'random_forest', 'logistic_regression']
    if cfg.get('model_type') not in valid_models:
        print(f"[ERROR] model_type must be one of {valid_models}")
        sys.exit(1)

    # ensure search_space provided
    if 'search_space' not in cfg or not isinstance(cfg['search_space'], dict):
        print("[ERROR] search_space must be defined as a dict in config")
        sys.exit(1)

    return cfg

def init_spark(app_name="train"):
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
    Returns X_tt, y_tt, X_oot, y_oot as pandas DataFrames.
    """
    # load label store
    folder_path = 'datamart/gold/label_store/'
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    label_store_sdf = spark.read.option('header', 'true').parquet(*files_list)

    # filter by overall date window
    labels_sdf = label_store_sdf.filter(
        (col('snapshot_date') >= config['train_test_start_date']) &
        (col('snapshot_date') <= config['oot_end_date'])
    )

    # load feature store
    feature_location = 'datamart/gold/feature_store'
    features_files_list = [
        os.path.join(feature_location, os.path.basename(f))
        for f in glob.glob(os.path.join(feature_location, '*'))
    ]
    features_store_sdf = spark.read.option('header', 'true').parquet(*features_files_list)

    # filter features by same window
    features_sdf = features_store_sdf.filter(
        (col('snapshot_date') >= config['train_test_start_date']) &
        (col('snapshot_date') <= config['oot_end_date'])
    )

    # convert to pandas and sort
    y_df = labels_sdf.toPandas().sort_values(by='customer_id')
    X_df = features_sdf.toPandas().sort_values(by='customer_id')

    # datetime conversion
    X_df['snapshot_date'] = pd.to_datetime(X_df['snapshot_date'])
    y_df['snapshot_date'] = pd.to_datetime(y_df['snapshot_date'])

    # simulate inner join
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


def split_train_test(X_tt, y_tt, ratio):
    """
    Performs train/test split and returns scaled arrays plus scaler.
    """
    X = X_tt.drop(columns=['customer_id', 'snapshot_date'])
    y = y_tt['label']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=ratio, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_te_s, y_tr, y_te, scaler


def get_model_and_params(model_type, search_space):
    """
    Returns estimator instance and param_distributions.
    """
    if model_type == 'xgboost':
        est = xgb.XGBClassifier(eval_metric='auc', random_state=42)
    elif model_type == 'random_forest':
        est = RandomForestClassifier(random_state=42)
    elif model_type == 'logistic_regression':
        est = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return est, search_space


def train_and_tune(X_tr_s, y_tr, estimator, params):
    """
    Runs RandomizedSearchCV and returns best estimator.
    """
    scorer = make_scorer(roc_auc_score)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        scoring=scorer,
        n_iter=100,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    search.fit(X_tr_s, y_tr)
    return search.best_estimator_, search.best_params_


def evaluate(model, X_s, y, label):
    """
    Evaluate model performance and output AUC score.
    """
    prob = model.predict_proba(X_s)[:, 1]
    auc = roc_auc_score(y, prob)
    print(f"{label} AUC: {auc:.4f}")
    return auc


def save_artefact(model, scaler, config, best_params, stats):
    """
    Save artefact elements.
    """
    artefact = {
        'model': model,
        'preprocessing_transformers': {'stdscaler': scaler},
        'model_version': 'credit_model_' + config['start_date_str'].replace('-', '_'),
        'config': config,
        'hp_params': best_params,
        'results': stats
    }
    os.makedirs('model_bank', exist_ok=True)
    path = os.path.join('model_bank', artefact['model_version'] + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(artefact, f)
    print(f"Saved model artefact at {path}")
    return artefact


def log_metrics(artefact):
    """
    Simple model metrics logging using MLflow. Using port 8000 due to the Apple Corporation using port 5000 on local machin
    for an unrelated system.
    """
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:8000'))
    mlflow.set_experiment(artefact['model_version'])
    with mlflow.start_run(run_name=artefact['model_version']):
        for k, v in artefact['results'].items():
            mlflow.log_metric(k, v)

def store_performance_table(artefact, X_tr_s, y_tr, X_te_s, y_te, X_oot_s, y_oot):
    """
    Read (or create) a CSV in datamart/gold/model_performance.csv,
    append this run’s metrics (AUC, precision, recall, F1 on train/test/OOT)
    and write it back.
    """
    perf_dir = 'datamart/gold'
    perf_file = os.path.join(perf_dir, 'model_performance.csv')

    # compute additional metrics
    y_tr_pred = artefact['model'].predict(X_tr_s)
    y_te_pred = artefact['model'].predict(X_te_s)
    y_oot_pred = artefact['model'].predict(X_oot_s)

    row = {
        'model_version': artefact['model_version'],
        'run_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_start_date': artefact['config']['train_test_start_date'],
        'data_end_date': artefact['config']['train_test_end_date'],
        'auc_train': artefact['results']['auc_train'],
        'precision_train': precision_score(y_tr, y_tr_pred),
        'recall_train':    recall_score(y_tr, y_tr_pred),
        'f1_train':        f1_score(y_tr, y_tr_pred),
        'auc_test':  artefact['results']['auc_test'],
        'precision_test':  precision_score(y_te, y_te_pred),
        'recall_test':     recall_score(y_te, y_te_pred),
        'f1_test':         f1_score(y_te, y_te_pred),
        'auc_oot':   artefact['results']['auc_oot'],
        'precision_oot':   precision_score(y_oot['label'], y_oot_pred),
        'recall_oot':      recall_score(y_oot['label'], y_oot_pred),
        'f1_oot':          f1_score(y_oot['label'], y_oot_pred),
    }

    # load existing or start new
    if os.path.exists(perf_file):
        df = pd.read_csv(perf_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    os.makedirs(perf_dir, exist_ok=True)
    df.to_csv(perf_file, index=False)
    print(f"[INFO] Appended performance to {perf_file}")


def main(config_path):
    cfg = get_config(config_path)
    spark = init_spark()
    X_tt, y_tt, X_oot, y_oot = preprocess_data(cfg, spark)

    # split train/test
    X_tr_s, X_te_s, y_tr, y_te, scaler = split_train_test(X_tt, y_tt, cfg['train_test_ratio'])

    # get model + params
    estimator, param_dist = get_model_and_params(cfg['model_type'], cfg['search_space'])

    # tune
    best_model, best_params = train_and_tune(X_tr_s, y_tr, estimator, param_dist)

    # evaluate
    stats = {}
    stats['auc_train'] = evaluate(best_model, X_tr_s, y_tr, 'Train')
    stats['auc_test'] = evaluate(best_model, X_te_s, y_te, 'Test')
    X_oot_s = scaler.transform(X_oot.drop(columns=['customer_id', 'snapshot_date']))
    stats['auc_oot'] = evaluate(best_model, X_oot_s, y_oot['label'], 'OOT')

    artefact = save_artefact(best_model, scaler, cfg, best_params, stats)
    store_performance_table(artefact, X_tr_s, y_tr, X_te_s, y_te, X_oot_s, y_oot)
    log_metrics(artefact)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_train.json')
    args = parser.parse_args()
    main(args.config)
