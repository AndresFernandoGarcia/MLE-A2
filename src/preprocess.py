import argparse
from datetime import datetime
import pyspark
from tqdm import tqdm
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
from utils.data_processing_gold_table import process_gold_table

def get_config(config_path):   
    try: 
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        return cfg['start_date'], cfg['end_date'], cfg['stage']
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)

def generate_first_of_month_dates(start_date_str, end_date_str):
    """ 
    Generate the dates in which the bronze, silver, and gold tables will be preprocesed for.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current_date = datetime(start_date.year, start_date.month, 1)

    dates = []
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    return dates

def run_bronze(dates_str_lst, spark):
    """ 
    Bronze table preprocessing.
    """ 
    print("Building bronze tables...")
    bronze_directory = "datamart/bronze"
    os.makedirs(bronze_directory, exist_ok=True)
    for name, path in [
        ('clickstream', 'data/feature_clickstream.csv'),
        ('attributes', 'data/features_attributes.csv'),
        ('financials', 'data/features_financials.csv'),
        ('lms', 'data/lms_loan_daily.csv')
    ]:
        for date_str in tqdm(dates_str_lst, desc=f"Processing {name}"):
            process_bronze_table(name, path, bronze_directory, date_str, spark)

def run_silver(dates_str_lst, spark):
    """ 
    Silver table preprocessing.
    """ 
    print("Building silver tables...")
    bronze_directory = "datamart/bronze"
    silver_directory = "datamart/silver"
    os.makedirs(silver_directory, exist_ok=True)
    for name in ['clickstream', 'attributes', 'financials', 'lms']:
        for date_str in tqdm(dates_str_lst, desc=f"Processing {name}"):
            process_silver_table(name, bronze_directory, silver_directory, date_str, spark)

def run_gold(dates_str_lst, spark):
    """ 
    Gold table preprocessing.
    """ 
    print("Building gold tables...")
    silver_directory = "datamart/silver"
    gold_directory = "datamart/gold"
    os.makedirs(gold_directory, exist_ok=True)
    process_gold_table(silver_directory, gold_directory, dates_str_lst, spark)

def main(config):
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    start_date, end_date, stage = get_config(config)
    print(f"Start Date: {start_date}, End Date: {end_date}, Preprocess up to stage: {stage}")
    dates_str_lst = generate_first_of_month_dates(start_date, end_date)

    if stage == 'bronze':
        run_bronze(dates_str_lst, spark)
    elif stage == 'silver':
        run_bronze(dates_str_lst, spark)
        run_silver(dates_str_lst, spark)
    elif stage == 'gold':
        run_bronze(dates_str_lst, spark)
        run_silver(dates_str_lst, spark)
        run_gold(dates_str_lst, spark)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/default_preprocess.json", help="Path to preprocess config JSON")
    args = parser.parse_args()
    main(args.config)