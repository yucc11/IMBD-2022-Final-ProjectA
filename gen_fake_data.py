
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import datetime

from argparse import Namespace


def _gen_fake_acch(data_dir="./data/raw/train1/", fake_type="train2", save_dir="./data/fake_projectA/train2", fname="accumulation_hour2.csv"):
    
    df_acch = pd.read_csv(Path(data_dir) / "accumulation_hour1.csv", parse_dates=['date'])
    if fake_type == "train2":
        end_date = datetime.datetime(2022, 6, 2)
        days = 28
    else: # test
        end_date = datetime.datetime(2022, 6, 30)
        days = 56
    
    df_acch["date"] = end_date  
    tar = df_acch[df_acch.accumulation_hour  > 0] 
    fake_slope = np.random.rand(len(tar))
    df_acch.loc[tar.index, "accumulation_hour"] += days*24*fake_slope
    df_acch["accumulation_hour"] = df_acch["accumulation_hour"].apply(lambda x: round(x))
    df_acch.to_csv(Path(save_dir) / fname, index=False)
    print(f"File `{Path(save_dir) / fname}` saved.")

def _gen_fake_anomaly(data_dir="./data/raw/train1/"):
    df_tr = pd.read_csv(Path(data_dir) / "anomaly_train1.csv", parse_dates=['date'])

    tar = df_tr[df_tr.date >= datetime.datetime(2022, 4, 7)]
    tar["date"] += datetime.timedelta(days=28)
    fname = "./data/fake_projectA/train2/anomaly_train2.csv"
    tar.to_csv(fname, index=False)
    print(f"File `{fname}` saved.")


def main(args: Namespace) -> None:
    
    _gen_fake_acch(data_dir=args.data_dir, fake_type="train2", save_dir="./data/fake_projectA/train2/", fname="accumulation_hour2.csv")
    _gen_fake_acch(data_dir=args.data_dir, fake_type="train3", save_dir="./data/fake_projectA/test/", fname="accumulation_hour3.csv")

    _gen_fake_anomaly(data_dir=args.data_dir)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="./data/raw/train1/", help="raw data dir"
    )
    args = parser.parse_args()
        
    main(args)