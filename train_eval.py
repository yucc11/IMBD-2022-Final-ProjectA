"""Main script for training and evaluation.

"""
import os
import sys
import pickle
import copy
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import argparse
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sklearn.model_selection import GroupKFold #, KFold, StratifiedKFold
from modeling.build import build_models
from utils.utils import seed_all
seed_all(42)

def _predict(model, x: pd.DataFrame) -> np.ndarray:
        """Do inference with the well-trained estimator.

        Parameters:
            model: well-trained estimator used to do inference
            x: data to predict on

        Return:
            y_pred: predicting results
        """
        y_pred = model.predict(x)

        return y_pred

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def _get_feat_imp(model, feat_names: List[str], imp_type: str) -> pd.DataFrame:
    """Generate and return feature importance DataFrame.

    Parameters:
        model: well-trained estimator
        feat_names: list of feature names
        imp_type: how the feature importance is calculated

    Return:
        feat_imp: feature importance
    """
    feat_imp = pd.DataFrame(feat_names, columns=["feature"])

    # if is_gbdt_instance(model, "lgbm"):
    feat_imp[f"importance_{imp_type}"] = model.booster_.feature_importance(imp_type)
    # elif is_gbdt_instance(model, ("xgb", "cat")):
    #     feat_imp[f"importance_{imp_type}"] = model.feature_importances_

    return feat_imp

def print_logs(logs):
    for log in logs:
        print(log)

def post_process(oof, clip=True):
    """"cilp negative prediction to 0"""
    df_oof = pd.DataFrame(oof)
    df_oof.loc[df_oof[0] < 0] = 0
    
    # clip prediction val < 0.5 to zero
    if clip:
        df_oof.loc[df_oof[0] < 0.5] = 0
    return df_oof[0].to_numpy()

def train_eval(models, cv, df, feat_cols, cat_cols, model_cfg):
    
    # cv
    oof = np.zeros(len(df))
    oof_scores = []
    logs = []
    imp = []
    val_feat_fold = []
    for ifold, (tr_idx, val_idx) in enumerate(cv.split(df, df["anomaly_total_number"], df["fold"])):# df["oven_layer_id"])):
        # Prepare train, val data
        tr_set, ts_set = df.iloc[tr_idx], df.iloc[val_idx]
        X_tr, y_tr = tr_set[feat_cols], tr_set["anomaly_total_number"]
        X_val, y_val = ts_set[feat_cols], ts_set["anomaly_total_number"]

        val_feat_fold += [ts_set.fold.unique()] # [X_val.index[0]]

        # Setup fit parameters
        fit_params_ifold = copy.copy(model_cfg["fit_params"])
        fit_params_ifold["eval_set"] = [(X_tr, y_tr), (X_val, y_val)]
        fit_params_ifold["categorical_feature"] = cat_cols # ["oven_id"]

        # Train the model
        models[ifold].fit(X_tr, y_tr, **fit_params_ifold)
        oof[val_idx] = _predict(models[ifold], X_val)
        
        # Evaluation
        tr_score = rmse(y_tr, _predict(models[ifold], X_tr))
        oof_score = rmse(y_val, oof[val_idx])       
        log_str = f"(RMSE fold {ifold}) |  Train: {tr_score} | oof: {oof_score}"       
        print(log_str)
        logs.append(log_str)
        oof_scores.append(oof_score)
        
        # Feature importance
        imp.append(_get_feat_imp(models[ifold], feat_cols, imp_type="gain"))

    final_oof_score = rmse(df["anomaly_total_number"], oof)
    final_post_oof_score = rmse(df["anomaly_total_number"], post_process(oof))
    
    logs.append(f"Fianl oof: {final_oof_score}")
    logs.append(f"Fianl oof (post process): {final_post_oof_score}")

    # print(logs)
    # print(f"Fianl oof: {final_oof_score}")
    # print(f"Fianl oof (post process): {final_post_oof_score}")

    return logs, oof, imp, val_feat_fold

def dump_model(model, model_type: str, mid: int, seed: int) -> None:
    """Dump estimator to corresponding path.

    Parameters:
        model: well-trained estimator
        model_type: type of the model, the choices are as follows:
            {"fold", "whole"}
        mid: identifer of the model

    Return:
        None
    """
    DUMP_PATH = Path("./output/")
    # model_file_prefix = "fold" if model_type == "fold" else "seed"

    # with open(DUMP_PATH / model_type / f"{model_file_prefix}{mid}.pkl", "wb") as f:
    with open(DUMP_PATH / model_type / f"seed{seed}_fold{mid}.pkl", "wb") as f:
        pickle.dump(model, f)

def main(args: Namespace) -> None:
    DATA_PATH = Path("./data/processed/")
    cfg_path = "./config/model/lgbm_template.yaml"

    # Load processed data
    df_train = pd.read_csv(DATA_PATH / args.proc_file)
    print(f"Load processed file: {args.proc_file}")
    print(f"Shape: {df_train.shape}")

    # Primary key
    df_train["oven_layer_id"] = df_train.apply(lambda x: x.oven_id + "_" + str(x.layer_id), axis=1)

    # Encode categorical feature
    idx_oid = pd.DataFrame([[i, oid] for i, oid in enumerate(df_train.oven_id.unique())])
    idx2oid = idx_oid.set_index(0).to_dict()[1]
    oid2idx = idx_oid.set_index(1).to_dict()[0]
    df_train["oven_id"] = df_train["oven_id"].map(oid2idx)
    with open(DATA_PATH/"oid2idx.pkl", "wb") as f:
        pickle.dump(oid2idx, f)


    # Load config
    with open(cfg_path, "r") as f:
        model_cfg = yaml.full_load(f)

    # Get all feature columns
    feat_cols = df_train.columns.to_list()
    feat_cols.remove("S20_A_temperature")
    feat_cols.remove("S20_B_temperature")
    feat_cols.remove("anomaly_total_number") # y
    feat_cols.remove("oven_layer_id")
    feat_cols.remove("fold")

    # feat_cols.remove("fold_diff_acc_hour")
    # feat_cols.remove("ol_diff_acc_hour")

    # TODO:
    # feat_cols.remove("fold_end_acc_hr")
    # feat_cols.remove("power_s1")
    # feat_cols.remove("power_s2")

    # Get categorical feature columns
    cat_cols = ["oven_id"]
    n_seeds = int(args.n_seeds)
    
    # 10 seed perf
    rst = []
    oof_scores = []
    for i, seed in enumerate(np.random.randint(20000, size=n_seeds).tolist()):
        model_cfg["model_params"]["random_state"] = seed

        cv = GroupKFold(n_splits=5)
        models = build_models("lgbm", model_cfg["model_params"], cv.get_n_splits())

        logs, oof, imp, val_feat_fold = train_eval(models, cv, df_train, feat_cols, cat_cols, model_cfg)
        rst.append({"seed": seed, "perf": logs})
        oof_score = rmse(df_train["anomaly_total_number"], post_process(oof))
        oof_scores.append(oof_score)
        # save model
        for fold, model in enumerate(models):
            dump_model(model, model_type="fold", mid=fold, seed=i)


    print(f"10 seed avg: {sum(oof_scores)/10}")
    
    # models = build_models("lgbm", model_cfg["model_params"], cv.get_n_splits())
    # logs, oof, imp, val_feat_fold = train_eval(models, cv, df_train, feat_cols, cat_cols, model_cfg)
    
    print(f"Logs\n {logs}")
    print(f"Feat num: {len(feat_cols)}")
    print(f"Feats: \n{feat_cols}")
    feat_cols = pd.DataFrame(feat_cols)
    feat_cols.to_csv("./data/processed/feat_cols.csv", index=False)
    print(val_feat_fold)
    # for im in imp:
    #     print(im.sort_values("importance_gain"))

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proc-file", default="train_1_2.csv", help="processed file name train_1_2.csv or train_1.csv"
    )
    parser.add_argument(
        "--n-seeds", default="10", help=""
    )
    args = parser.parse_args()
    main(args)

# save as
# models/seed{sid}_fold{fid}.pkl