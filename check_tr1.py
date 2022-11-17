
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

def _load_tr1_data(folder="/TOPIC/projectA/train1/"):
    RAW_DATA_PATH = Path(folder)
    df_tr = pd.read_csv(RAW_DATA_PATH / "anomaly_train1.csv", parse_dates=['date'])
    df_acch = pd.read_csv(RAW_DATA_PATH / "accumulation_hour1.csv", parse_dates=['date'])
    df_cooler = pd.read_csv(RAW_DATA_PATH / "cooler.csv")
    df_power = pd.read_csv(RAW_DATA_PATH / "power.csv")
    return df_tr, df_acch, df_cooler, df_power 

def main(args: Namespace) -> None:
    df_tr, df_acch, df_cooler, df_power  =_load_tr1_data(folder=f"{args.data_dir}/train1/")
    
    # save file
    save_dir = Path("./data/raw/train1/")
    
    # df_tr = df_tr.sort_values(["date", "oven_id", "layer_id"])
    fpath = save_dir / "anomaly_train1.csv"
    df_tr.to_csv(fpath, index=False)
    print(f"File `{fpath}` saved.")

    fpath = save_dir / "accumulation_hour1.csv"
    df_acch.to_csv(fpath, index=False)
    print(f"File `{fpath}` saved.")

    fpath = save_dir / "cooler.csv"
    df_cooler.to_csv(fpath, index=False)
    print(f"File `{fpath}` saved.")

    fpath = save_dir / "power.csv"
    df_power.to_csv(fpath, index=False)
    print(f"File `{fpath}` saved.")
    


    
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="/TOPIC/projectA/", help="readonly dir."
    )
    args = parser.parse_args()

    # Launch main function
    main(args)

 