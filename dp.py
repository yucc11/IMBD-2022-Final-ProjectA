"""Main script for processing train1, train2 and testing data.

"""
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
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def _load_tr1_data(folder="./data/raw/train1/"):
    RAW_DATA_PATH = Path(folder)
    df_tr = pd.read_csv(RAW_DATA_PATH / "anomaly_train1.csv", parse_dates=['date'])
    df_acch = pd.read_csv(RAW_DATA_PATH / "accumulation_hour1.csv", parse_dates=['date'])
    df_cooler = pd.read_csv(RAW_DATA_PATH / "cooler.csv")
    df_power = pd.read_csv(RAW_DATA_PATH / "power.csv")
    return df_tr, df_acch, df_cooler, df_power 

def _load_tr2_data(folder="./data/raw/train2/"):
    RAW_DATA_PATH = Path(folder)
    df_tr2 = pd.read_csv(RAW_DATA_PATH / "anomaly_train2.csv", parse_dates=['date'])
    df_acch2 = pd.read_csv(RAW_DATA_PATH / "accumulation_hour2.csv", parse_dates=['date'])
    return df_tr2, df_acch2

def _load_test_data(folder="./data/raw/test/"):
    RAW_DATA_PATH = Path(folder)
    df_ts = pd.read_csv(RAW_DATA_PATH / "accumulation_hour3.csv", parse_dates=['date'])
    return df_ts

def preprocess_tr_data(df_tr, df_acch, job_type="tr1"):
    if job_type == "tr1":
        last_date = datetime.datetime(2022, 5, 4)
    else: # "tr2"
        last_date = datetime.datetime(2022, 6, 2)
    
    # get anomaly samples on last date
    tar = (df_tr[df_tr.date == last_date])
    # avg hour
    tar = tar.merge(df_acch, how="inner", on=["date", "oven_id", "layer_id"])
    tar["anomaly_accumulation_hour"] = (tar["anomaly_accumulation_hour"]  + tar["accumulation_hour"]) / 2

    # Merge 5/4 accumulation hour & train1 anomaly samples
    df_time = pd.concat([df_acch.rename(columns={"accumulation_hour": "anomaly_accumulation_hour"}), df_tr[["date", "oven_id", "layer_id", "anomaly_accumulation_hour", "anomaly_total_number"]]], ignore_index=True)
    df_time = df_time.sort_values(["oven_id", "layer_id", "date"]).reset_index(drop=True)
    # remove acumulation hour of df_acch on 5/4 of tar
    df_du = df_time[df_time.date == last_date].groupby(["date", "oven_id", "layer_id"]).size().reset_index().rename(columns={0: "num_sample"})
    df_du = df_du[df_du.num_sample > 1]
    df_du["drop_row"] = 1
    df_time = df_time.merge(df_du[["date", "oven_id", "layer_id", "drop_row"]], how="outer", on=["date", "oven_id", "layer_id"])
    keep_row = []
    for i, row in df_time[df_time.drop_row == 1].iterrows():
        if row.anomaly_total_number == row.anomaly_total_number: # notna
            keep_row.append(i)

    df_time.loc[keep_row, "drop_row"] = 0
    df_time["drop_row"] = df_time["drop_row"].fillna(0)
    # drop samples
    df_time = df_time[df_time.drop_row == 0].drop("drop_row", axis=1).reset_index(drop=True)

    return df_time

def process_group(tar):
    """
    Parameters:
        tar: dataframe, group df

    """
    tar = tar.sort_values(["date"]).reset_index(drop=True)
    # previous anomaly sample
    tar["prev_date"] = [np.nan] + tar.date[:-1].tolist()
    tar["prev_hr"] = [0] + tar.anomaly_accumulation_hour[:-1].tolist()    
    # diff
    if len(tar) > 1:
        tar["diff_date"] = tar["date"] - tar["prev_date"]
    tar["diff_hr"] = tar["anomaly_accumulation_hour"] - tar["prev_hr"]

    # tar["diff_date_hr"] = tar["diff_date"].dt.days*24 
    # # assume the anomaly sample is recorded at the same timepoint (12:00 on each recorded date)
    
    # slope
    if len(tar) == 1:
        tar["slope"] = np.nan
    else:
        tar["slope"] = tar["diff_hr"] / (tar["diff_date"].dt.days*24)

    # calculate first date 
    fisrt_slope = 1
    tar.loc[0, "prev_date"] = tar.loc[0, "date"] - (datetime.timedelta(hours=int(tar.loc[0, "anomaly_accumulation_hour"]/fisrt_slope)))
    tar["max_diff_hr"] = (tar["date"] - tar["prev_date"] + datetime.timedelta(days=1)).dt.days*24
    
    # label first sample for ignoring slope
    tar["first_sample"] = 0
    tar.loc[0, "first_sample"] = 1
    
    return tar

def split_data(df_tr):
    """Split training data using time window=28days
    
    Parameters:
        df_tr: concat train1, train2

    Return df_tr, dt_end
    """
    # Train1
    start = datetime.datetime(2021, 12, 15)
    dt_start, dt_end = [], []
    f = []
    for i in range(5):
        dt_start += [start + datetime.timedelta(days=1)]
        df_tr.loc[df_tr["date"] > start , "fold"] = i
        start = start + datetime.timedelta(days=28)
        dt_end += [start]
        f += [i]

    # Train2
    for i in range(5, 6):
        dt_start += [start + datetime.timedelta(days=1)]
        df_tr.loc[df_tr["date"] > start , "fold"] = i
        print(f"Train2 num samples {len(df_tr.loc[df_tr['date'] > start , 'fold'])}")
        start = start + datetime.timedelta(days=29)
        dt_end += [start]
        f += [i]

    # Test # TODO: ? 
    for i in range(6, 7):
        dt_start += [start + datetime.timedelta(days=1)]
        df_tr.loc[df_tr["date"] > start , "fold"] = i 
        start = start + datetime.timedelta(days=28)
        dt_end += [start]
        f += [i]

    dt_start[0] = datetime.datetime(2021, 12, 27)
    for i in f:
        print(f"Fold {i}: {dt_start[i]} - {dt_end[i]}")
        print(f"{df_tr[df_tr.fold == i].date.min()} - {df_tr[df_tr.fold == i].date.max()}")
    return df_tr, dt_end

def gen_fold_data(df_tr, df_acch, num_fold=6, drop_fold0=False):
    """Generate 5 fold data for splited df. Fill 0 for no anomaly month.
    Parameters:
        df_tr: concat train1, train2
        num_fold: set 5 for train1, 6 for train1+train2

    Return df_train
    """
    # training format
    oven_ids = df_acch["oven_id"].unique().tolist()
    new = []
    for oid in oven_ids:        
        for lid in range(1, 20):
            for fold in range(num_fold):
                new.append([oid, lid, fold])

    df_train = pd.DataFrame(new, columns=["oven_id", "layer_id", "fold"])
    print(f"Gen fold data: {df_train.shape} | (950, 3)")

    # count anomaly total number for each fold
    df_tar = df_tr.groupby(["oven_id", "layer_id", "fold"]).agg({"anomaly_total_number": "sum"})
    df_train = df_train.merge(df_tar, how="outer", on=["oven_id", "layer_id", "fold"])
    df_train = df_train.fillna(0)
    # ignore fold 0
    if drop_fold0:
        df_train = df_train[df_train.fold > 0]
    # print(df_train.shape)
    return df_train

def gen_feats(df_proc, df_train, df_acch, df_cooler, df_power, dt_end, df_acch2=None):
    """
    Parameters:
        df_proc: source dataframe for feature generation
        df_train: result 
        ### df_acch: df_acch2 for data_type == 'tr2'
    Return:
        df_train
    """
    ## Feat -- oven layer fold
    # `fold_avg_slope`
    gps = df_proc.loc[df_proc.drop_slope == 0].groupby(["oven_id", "layer_id", "fold"])
    gps_avg_slope = gps["slope"].mean().reset_index().rename(columns={"slope": "fold_avg_slope"})
    df_train = df_train.merge(gps_avg_slope, on=["oven_id", "layer_id", "fold"], how="outer")
    # `fold_diff_acc_hour`: Diff acc hour
    gps_diff_acc_hour = (gps["slope"].mean()*28*24).reset_index().rename(columns={"slope": "fold_diff_acc_hour"})
    df_train = df_train.merge(gps_diff_acc_hour, on=["oven_id", "layer_id", "fold"], how="outer")
    
    # fill na with previous fold avg slope
    prev_fold_slope = []
    tr_gps = df_train.sort_values(["oven_id", "layer_id", "fold"]).groupby(["oven_id", "layer_id"])
    for i, gp in tr_gps:
        prev_fold_slope += [np.nan] + gp["fold_avg_slope"][:-1].tolist()
    df_train["prev_fold_slope"] = prev_fold_slope
    for i, row in df_train.iterrows():
        if row.fold_avg_slope != row.fold_avg_slope:
            df_train.loc[i, "fold_avg_slope"] = df_train.loc[i, "prev_fold_slope"]
    df_train = df_train.drop("prev_fold_slope", axis=1)

    
    ## Feat -- oven layer
    # `ol_avg_slope``: Avg group slope, ignore outliers
    gps = df_proc.loc[df_proc.drop_slope == 0].groupby(["oven_id", "layer_id"])
    gps_avg_slope = gps["slope"].mean().reset_index().rename(columns={"slope": "ol_avg_slope"})
    df_train = df_train.merge(gps_avg_slope, on=["oven_id", "layer_id"], how="outer")
    # `ol_diff_acc_hour``: Diff acc hour
    gps_diff_acc_hour = (gps["slope"].mean()*28*24).reset_index().rename(columns={"slope": "ol_diff_acc_hour"})
    df_train = df_train.merge(gps_diff_acc_hour, on=["oven_id", "layer_id"], how="outer")

    ## Feat -- oven
    # `oven_avg_slope`
    gps = df_proc.loc[df_proc.drop_slope == 0].groupby(["oven_id"])
    gps_avg_slope = gps["slope"].mean().reset_index().rename(columns={"slope": "oven_avg_slope"})
    df_train = df_train.merge(gps_avg_slope, on=["oven_id"], how="outer")

    ## Feat -- `end_acc_hr` from df_acch and df_acch2 ##
    df_train = df_train.merge(df_acch.rename(columns={"accumulation_hour": "end_acc_hr"}), how="outer", on=["oven_id", "layer_id"])
    df_train = df_train.drop("date", axis=1)
    if df_acch2 is not None:
        df_train = df_train.merge(df_acch2.rename(columns={"accumulation_hour": "end_acc_hr2"}), how="outer", on=["oven_id", "layer_id"])
        df_train = df_train.drop("date", axis=1)

    ## Feat -- `fold_end_acc_hr` 
    # `anomaly_accumulation_hour` at the end of fold i 用每個爐層下單一fold中最後一筆資料計算fold最後一天時的運行時數
    gps = df_proc.loc[df_proc.drop_slope == 0].groupby(["oven_id", "layer_id", "fold"])
    df_new = []
    for gp_idx, gp in gps:
        last_row = gp.loc[gp.index[-1]]
        if (last_row.date == datetime.datetime(2022, 5, 4)) or (last_row.date == datetime.datetime(2022, 6, 2)):
            acc_hr = last_row.anomaly_accumulation_hour
            df_new.append([last_row.oven_id, last_row.layer_id, last_row.fold, acc_hr])
        elif last_row.fold < 5:
            slope = last_row.slope
            acc_hr = last_row.anomaly_accumulation_hour + slope * (dt_end[int(last_row.fold)] - last_row.date).days*24
            df_new.append([last_row.oven_id, last_row.layer_id, last_row.fold, acc_hr])
    df_new = pd.DataFrame(df_new)
    df_new = df_new.rename(columns={0: "oven_id", 1: "layer_id", 2: "fold", 3: "fold_end_acc_hr"})
    df_train = df_train.merge(df_new, how="outer", on=["oven_id", "layer_id", "fold"])

    # ## Feat -- `end_acc_hr` from df_acch ##
    # df_train = df_train.merge(df_acch.rename(columns={"accumulation_hour": "end_acc_hr"}), how="outer", on=["oven_id", "layer_id"])
    # df_train = df_train.drop("date", axis=1)
    
    ## Feat -- `fold_end_acc_hr` fill Nan ##
    # 1. tr1:
    # fold_end_acc_hr == nan時可能為該爐層在fold中沒有異常資料可以用來計算運行時數 改用5/4運行時數＆ol_avg_slope回推fold最後一天該爐層的運行時數
    
    idx, tar_val = [], []
    for i, row in df_train[df_train.fold_end_acc_hr.isna()].iterrows():
        if row.fold < 5:
            val = row.end_acc_hr - row.ol_avg_slope * (dt_end[4] - dt_end[row.fold]).days*24
            # idx.append(i)
            # tar_val.append(val)
        elif row.fold == 5:
            val = row.end_acc_hr
        elif row.fold == 6:
            val = row.end_acc_hr2
        idx.append(i)
        tar_val.append(val)
    df_train.loc[idx, "fold_end_acc_hr"] = tar_val
    # clip negative value to 0
    df_train.loc[df_train.fold_end_acc_hr < 0, "fold_end_acc_hr"] = 0
    # print(df_train[df_train.fold_end_acc_hr.isna()].shape)
    
    idx, tar_val = [], []
    for i, row in df_train[df_train.fold_end_acc_hr.isna()].iterrows():
        if row.fold < 5:
            val = row.end_acc_hr - row.oven_avg_slope * (dt_end[4] - dt_end[row.fold]).days*24
            idx.append(i)
            tar_val.append(val)
    df_train.loc[idx, "fold_end_acc_hr"] = tar_val
    # clip negative value to 0
    df_train.loc[df_train.fold_end_acc_hr < 0, "fold_end_acc_hr"] = 0
    # 2. tr2: TODO:
    # # fold_end_acc_hr == nan時可能為該爐層在fold中沒有異常資料可以用來計算運行時數 改用6/2運行時數＆ol_avg_slope回推fold最後一天該爐層的運行時數
    # print(df_train.isna().sum())
    df_train = df_train.drop("end_acc_hr", axis=1)
    if df_acch2 is not None:
        df_train = df_train.drop("end_acc_hr2", axis=1)

    ## Feat: cooler feats
    df_cooler_T = df_cooler.T
    df_cooler_T.columns = df_cooler_T.iloc[0, :]
    df_cooler_T = df_cooler_T.iloc[1:, :]
    df_cooler_T = df_cooler_T.reset_index().rename(columns={"index": "oven_id"}) # oven2feat
    for col in df_cooler_T.columns.tolist()[1:-2]:
        df_cooler_T[col] = pd.to_numeric(df_cooler_T[col])
    df_train = pd.merge(df_train, df_cooler_T, how="outer", on=["oven_id"])

    ## Feat: `power_s1` `power_s2` 
    df_power["lower_bound"] = df_power["accumulation_hour"].apply(lambda x: int(x.split("-")[0]))
    df_power["upper_bound"] = df_power["accumulation_hour"].apply(lambda x: int(x.split("-")[1]))
    # s1 = set([1, 2, 60, 61, 62, 63, 121, 122])
    # s2 = set([i for i in range(1, 123)]).difference(s1)
    hr2power_s1 = df_power[["accumulation_hour", "power_setup(lamp_1_2_60_61_62_63_121_122)"]].set_index("accumulation_hour").to_dict()["power_setup(lamp_1_2_60_61_62_63_121_122)"]
    hr2power_s2 = df_power[["accumulation_hour", "power_setup(other_lamp)"]].set_index("accumulation_hour").to_dict()["power_setup(other_lamp)"]
    
    powers1, powers2 = [], []
    for i, row in df_train.iterrows():
        acch = row.fold_end_acc_hr
        if acch != acch:
                powers1.append(np.nan)
                powers2.append(np.nan)
        else:
            acch = round(acch)
            xrange = ""
            for lb, ub in zip(df_power["lower_bound"], df_power["upper_bound"]):
                i += 1
                if acch >= lb and acch <= ub:
                    xrange = f"{lb}-{ub}"
                    if (xrange in hr2power_s1) and (xrange in hr2power_s2):
                        powers1.append(hr2power_s1[xrange])
                        powers2.append(hr2power_s2[xrange])
                    break
            if xrange == "":
                print(f"Power Feat - row{i}: {acch}")
                powers1.append(np.nan)
                powers2.append(np.nan)
            
    df_train["power_s1"] = powers1
    df_train["power_s2"] = powers2
    return df_train
    
    
def _gen_train_data(data_type="tr2") -> None:
    """Generate training data.

    Parameters:
        data_type: 
            set "tr1" for processing train1 data. File will be save as ./data/processed/train_1.csv
            set "tr2" for processing train1 and train2 data. File will be save as ./data/processed/train_1_2.csv
    """
    
    # load raw data
    df_tr, df_acch, df_cooler, df_power = _load_tr1_data(folder="./data/raw/train1/")
    if data_type == "tr2":
        df_tr2, df_acch2 = _load_tr2_data(folder="./data/raw/train2/")
            
    # merge & revise acch on last date
    df_time = preprocess_tr_data(df_tr, df_acch, job_type="tr1")
    if data_type == "tr2":
        df_time2 = preprocess_tr_data(df_tr2, df_acch2, job_type="tr2")
        df_time = pd.concat([df_time, df_time2])

    # process group df
    df_proc = []
    gp = df_time.groupby(["oven_id", "layer_id"])
    gp_idx = gp.groups.keys()
    for gp_idx in gp.indices:
        tar = gp.get_group(gp_idx)
        tar = process_group(tar)
        df_proc.append(tar)
    df_proc = pd.concat(df_proc).reset_index(drop=True)

    # filter outliers
    df_proc["drop_slope"] = 0 # drop `slope`, for ovenlayer first sample: drop `slope` `prev_date` `max_diff_hr`,  
    df_proc.loc[df_proc.slope <= 0, "drop_slope"] = 1
    df_proc.loc[df_proc.diff_hr	> df_proc.max_diff_hr, "drop_slope"] = 1
    df_proc.loc[df_proc.prev_date > df_proc.date, "drop_slope"] = 1
    df_proc.loc[df_proc.slope != df_proc.slope, "drop_slope"] = 1
    df_proc.loc[df_proc.first_sample == 1, "drop_slope"] = 1

    # gen `fold`
    df_proc, dt_end = split_data(df_proc)
    #  train data format
    print(f"Generating Feats...")
    if data_type == "tr1":
        num_fold = 5
        print(f"num_fold: {num_fold}")    
    elif data_type == "tr2":
        num_fold = 6
        print(f"num_fold: {num_fold}")    
    df_train = gen_fold_data(df_proc, df_acch, num_fold, drop_fold0=False)## TODO: set cfg drop_fold0
    # gen "fold_avg_slope", "fold_diff_acc_hour", "ol_avg_slope", "ol_diff_acc_hour"
    if data_type == "tr1":
        df_train = gen_feats(df_proc, df_train, df_acch, df_cooler, df_power, dt_end)
    else:
        df_train = gen_feats(df_proc, df_train, df_acch, df_cooler, df_power, dt_end, df_acch2)
        
    # save file
    if data_type == "tr1":
        save_name = "train_1.csv"
    elif data_type == "tr2":
        save_name = "train_1_2.csv"
    fpath = f"./data/processed/{save_name}"

    df_train.to_csv(fpath, index=False)
    # print(df_train)
    print(df_train.isna().sum())
    print(f"Training data: {df_train.shape}. File saved at {fpath}")
    # test block
    # df_proc.to_csv("proc.csv", index=False)

def _gen_test_data(data_type="ts", ts_source="tr_12") -> None:
    """Generate input data for inference.
    
    Generated features: 
        `fold_avg_slope`
        `fold_diff_acc_hour`

        `ol_avg_slope`
        `ol_diff_acc_hour`

        `oven_avg_slope`
        `end_acc_hr` : del
        `fold_end_acc_hr`
        
        cooler (total: 64 feats)

        `power_s1`
        `power_s2`

    File will be save as ./data/processed/test.csv
    
    """
    # load raw 
    df_tr, df_acch, df_cooler, df_power  = _load_tr1_data(folder="./data/raw/train1")
    df_tr2, df_acch2 = _load_tr2_data(folder="./data/raw/train2/")
    df_ts = _load_test_data(folder="./data/raw/test/")
    
    # load processed data
    df_train = pd.read_csv("./data/processed/train_1.csv")
    # df_train2 = pd.read_csv("./data/processed/train_2.csv")
    tar1 = df_train.groupby(["oven_id", "layer_id"]).agg({"ol_avg_slope": "mean"}).reset_index()
    # tar2 = df_train2.groupby(["oven_id", "layer_id"]).agg({"ol_avg_slope": "mean"}).reset_index()
    df_ts = df_ts.merge(tar1, how="outer", on=["oven_id", "layer_id"])
    
    df_ts["oven_id_str"] = df_train["oven_id"]
    ## Features ##
    # fold feat
    df_ts["fold_avg_slope"] = (df_ts.accumulation_hour - df_acch2.accumulation_hour) / 28
    df_ts["fold_diff_acc_hour"] = df_ts["fold_avg_slope"]*28*24
    # TODO: ts_source=="tr_12" or ts_source="tr_1"
    # TODO: fill na or fold_avg slope < 0 with previous fold avg slope
    # TODO: 填完之後用斜率推算正確的累積運作時間(power!)


    if ts_source =="tr_12":
        # load train_1_2.csv
        df_train12 = pd.read_csv("./data/processed/train_1_2.csv")
        tar2 = df_train12.groupby(["oven_id", "layer_id"]).agg({"ol_avg_slope": "mean"}).reset_index()
        df_ts = df_ts.merge(tar2, how="outer", on=["oven_id", "layer_id"], suffixes=["", "_2"])

        # oven layer feat
        ts_slopes = []
        for i, row in df_ts.iterrows():
            slopes = [row.ol_avg_slope_2 for i in range(6)]
            slopes += [row.fold_avg_slope]
            slopes = [x for x in slopes if (x == x and x >= 0)]
            if len(slopes) == 0:
                ts_slopes += [np.nan]
            else:
                ts_slopes += [sum(slopes) / len(slopes)]

        df_ts["ol_avg_slope_3"] = ts_slopes
        df_ts = df_ts.drop(["ol_avg_slope", "ol_avg_slope_2"], axis=1)
    else: 
        # oven layer feat
        ts_slopes = []
        for i, row in df_ts.iterrows():
            slopes = [row.ol_avg_slope for i in range(5)]
            slopes += [row.fold_avg_slope]
            slopes = [x for x in slopes if (x == x and x >= 0)]
            if len(slopes) == 0:
                ts_slopes += [np.nan]
            else:
                ts_slopes += [sum(slopes) / len(slopes)]

        df_ts["ol_avg_slope_3"] = ts_slopes
        df_ts = df_ts.drop(["ol_avg_slope"], axis=1)

    df_ts = df_ts.rename(columns={"ol_avg_slope_3": "ol_avg_slope"})
    df_ts["ol_diff_acc_hour"] = df_ts["ol_avg_slope"]*28*24

    # oven feat
    gps_avg_slope = df_ts.groupby(["oven_id"]).agg({"ol_avg_slope": "mean"}).reset_index().rename(columns={"ol_avg_slope": "oven_avg_slope"})
    df_ts = df_ts.merge(gps_avg_slope, on=["oven_id"], how="outer")
    
    # Feat -- `fold_end_acc_hr` & `end_acc_hr` 
    df_ts["fold_end_acc_hr"] = df_ts["accumulation_hour"]
    # df_ts["end_acc_hr"] = df_ts["accumulation_hour"] 
    df_ts = df_ts.drop(["accumulation_hour"], axis=1)
    # feat cooler
    df_cooler_T = df_cooler.T
    df_cooler_T.columns = df_cooler_T.iloc[0, :]
    df_cooler_T = df_cooler_T.iloc[1:, :]
    df_cooler_T = df_cooler_T.reset_index().rename(columns={"index": "oven_id"}) # oven2feat
    for col in df_cooler_T.columns.tolist()[1:-2]:
        df_cooler_T[col] = pd.to_numeric(df_cooler_T[col])
    df_ts = df_ts.merge(df_cooler_T, how="outer", on=["oven_id"])

    # feat power
    df_power["lower_bound"] = df_power["accumulation_hour"].apply(lambda x: int(x.split("-")[0]))
    df_power["upper_bound"] = df_power["accumulation_hour"].apply(lambda x: int(x.split("-")[1]))

    # Add feature: Power
    # s1 = set([1, 2, 60, 61, 62, 63, 121, 122])
    # s2 = set([i for i in range(1, 123)]).difference(s1)
    hr2power_s1 = df_power[["accumulation_hour", "power_setup(lamp_1_2_60_61_62_63_121_122)"]].set_index("accumulation_hour").to_dict()["power_setup(lamp_1_2_60_61_62_63_121_122)"]
    hr2power_s2 = df_power[["accumulation_hour", "power_setup(other_lamp)"]].set_index("accumulation_hour").to_dict()["power_setup(other_lamp)"]

    powers1, powers2 = [], []
    for i, row in df_ts.iterrows():
        acch = row.fold_end_acc_hr
        if acch != acch:
                powers1.append(np.nan)
                powers2.append(np.nan)
        else:
            xrange = ""
            for lb, ub in zip(df_power["lower_bound"], df_power["upper_bound"]):
                if acch >= lb and acch <= ub:
                    xrange = f"{lb}-{ub}"
                    if (xrange in hr2power_s1) and (xrange in hr2power_s2):
                        powers1.append(hr2power_s1[xrange])
                        powers2.append(hr2power_s2[xrange])

                    break
            if xrange == "":
                print(f"Power Feat - row{i}: {acch}")
                powers1.append(np.nan)
                powers2.append(np.nan)
            
    df_ts["power_s1"] = powers1
    df_ts["power_s2"] = powers2

    print(f"df_ts shape: {df_ts.shape} | (190, 74).")
    df_ts = df_ts.replace(np.inf, np.nan)
    df_ts = df_ts.replace(-np.inf, np.nan)
    # save file
    df_ts.to_csv("./data/processed/test.csv", index=False)
    return

def main(args: Namespace) -> None:
    if args.data_type == "ts":
        _gen_test_data(data_type=args.data_type, ts_source=args.ts_source)

    else: # tr1 or tr2
       _gen_train_data(data_type=args.data_type)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-type", default="tr2", help="[tr1, tr2, ts]"
    )
    parser.add_argument(
        "--ts-source", default="tr_12", help="set for test data. tr_12 or tr_1"
    )
    args = parser.parse_args()

    # Launch main function
    main(args)

# save path
# ./data/processed/test.csv
# ./data/processed/train_1.csv
# ./data/processed/train_1_2.csv


