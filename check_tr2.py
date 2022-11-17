
import os
import sys
import copy
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import argparse

from argparse import Namespace

pd.options.display.max_rows = None
pd.options.display.max_columns = None

def _load_tr2_data(folder="/TOPIC/projectA/train2/"):
    RAW_DATA_PATH = Path(folder)
    df_tr2 = pd.read_csv(RAW_DATA_PATH / "anomaly_train2.csv", parse_dates=['date'])
    df_acch2 = pd.read_csv(RAW_DATA_PATH / "accumulation_hour2.csv", parse_dates=['date'])
    return df_tr2, df_acch2


def _load_test_data(folder="/TOPIC/projectA/test/"):
    RAW_DATA_PATH = Path(folder)
    df_ts = pd.read_csv(RAW_DATA_PATH / "accumulation_hour3.csv", parse_dates=['date'])
    return df_ts

def main(args: Namespace) -> None:

    df_tr2, df_acch2 = _load_tr2_data(folder=f"{args.data_dir}/train2/")
    df_ts = _load_test_data(folder=f"{args.data_dir}/test/")
    
    # check duplicate anomaly samples
    tar = df_tr2.groupby(["date", "oven_id", "layer_id"]).size().reset_index().rename(columns={0: "num_sample"})
    if len(tar[tar.num_sample > 1]) > 0:
        print(f"\n df_tr2 | Num of samples > 1: \n{tar[tar.num_sample > 1]}")

    # check date in df_acch  
    tr2_fixed = df_acch2[df_acch2.date != datetime.datetime(2022, 6, 2)] 
    if len(tr2_fixed) > 0:
        print(f"\n df_acch2 | date != 6/2: \n{tr2_fixed}")
        df_acch2.loc[df_acch2.date != datetime.datetime(2022, 6, 2), "date"] = datetime.datetime(2022, 6, 2)

    # check date in df_acch  
    ts_fixed = df_ts[df_ts.date != datetime.datetime(2022, 6, 30)] 
    if len(ts_fixed) > 0:
        print(f"\n df_ts | date != 6/30: \n{ts_fixed}")
        df_ts.loc[df_ts.date != datetime.datetime(2022, 6, 30), "date"] = datetime.datetime(2022, 6, 30)


    # save file
    df_tr2 = df_tr2.sort_values(["date", "oven_id", "layer_id"])
    fpath = "./data/raw/train2/anomaly_train2.csv"
    df_tr2.to_csv(fpath, index=False)
    print(f"File `{fpath}` saved.")

    df_acch2 = df_acch2.sort_values(["date", "oven_id", "layer_id"])
    fpath = "./data/raw/train2/accumulation_hour2.csv"
    df_acch2.to_csv(fpath, index=False)
    print(f"File `{fpath}` saved.")

    df_ts = df_ts.sort_values(["date", "oven_id", "layer_id"])
    fpath = "./data/raw/test/accumulation_hour3.csv"
    df_ts.to_csv(fpath, index=False)
    print(f"File `{fpath}` saved.")
    
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="/TOPIC/projectA/", help="readonly dir. use `./data/fake_projectA/` for testing."
    )
    args = parser.parse_args()

    # Launch main function
    main(args)

 