"""Main script for inference."""
import os
import pickle

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from argparse import Namespace

def _load_test_data():
    # Load processed data
    fname = "./data/processed/test.csv"
    df_test = pd.read_csv(fname)
    feat_cols = pd.read_csv(Path("./data/processed/") / "feat_cols.csv")
    print(f"Load processed file: {fname}")
    print(f"Shape: {df_test.shape}")
    print(feat_cols)
    return df_test, feat_cols.loc[:, "0"]

def _post_process(y_pred, clip=True):
    """"cilp negative prediction to 0"""
    df_oof = pd.DataFrame(y_pred)
    df_oof.loc[df_oof[0] < 0] = 0
    
    # clip prediction val < 0.5 to zero
    if clip:
        df_oof.loc[df_oof[0] < 0.5] = 0
    return df_oof[0].to_numpy()

def _run_infer(X: pd.DataFrame, model_dir: str, n_seeds: int, n_folds: int) -> np.ndarray:
    """Run inference and return predicting results.

    For a single inference process, predictions are averaged over all
    models obtained in single CV round (i.e., #models=n_seeds*n_folds).

    Parameters:
        X: features to feed into pre-trained models
        model_dir: directory of dumped pre-trained models
        n_seeds: number of seeds used in a single CV round
        n_folds: number of folds in a single data splitting

    Return:
        y_pred: average predicting result
    """
    n_models = n_seeds * n_folds
    y_pred = np.zeros(len(X))
    for sid in range(n_seeds):
        for fid in range(n_folds):
            model_path = os.path.join(model_dir, f"seed{sid}_fold{fid}.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            y_pred += model.predict(X) / n_models

    y_pred = _post_process(y_pred, clip=False)
    return y_pred


def _save_prediction(X, y_pred, save_dir, fname) -> None:
    """ Save prediction result as sumbmission format.

    Parameters:
        X: features to feed into pre-trained models
        y_pred: prediction result
    """
    X["y_pred"] = y_pred
    X["id"] = X.apply(lambda x: x.oven_id_str + "-" + str(x.layer_id), axis=1)
    X.sort_values(["oven_id_str", "layer_id"])[["id", "y_pred"]].to_csv(Path(save_dir) / fname, index=False)
    print(f"File `{save_dir} / {fname}` saved.")

def main(args: Namespace) -> None:

    X, feat_cols = _load_test_data()
    print(f"feat_cols {feat_cols}")
    print(f"Feat num {len(feat_cols)}")
    print(X[feat_cols].dtypes)

    # Encode categorical feature
    # idx_oid = pd.DataFrame([[i, oid] for i, oid in enumerate(X.oven_id.unique())])
    # idx2oid = idx_oid.set_index(0).to_dict()[1]
    # oid2idx = idx_oid.set_index(1).to_dict()[0]
    with open("./data/processed/oid2idx.pkl", "rb") as f:
        oid2idx = pickle.load(f)
    X["oven_id"] = X["oven_id"].map(oid2idx)

    y_pred = _run_infer(
                X[feat_cols],
                args.model_dir,
                n_seeds=10,
                n_folds=5,
           )
    
    _save_prediction(X, y_pred, args.save_dir, args.save_fname)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--input-fname", default="test.csv", help="infer file name"
    # )
    parser.add_argument(
        "--model-dir", default="./output/fold", help=""
    )
    parser.add_argument(
        "--save-dir", default="", help="./output/"
    )
    parser.add_argument(
        "--save-fname", default="projectA_prediction.csv", help=""
    )
    args = parser.parse_args()
    
    # Run main function
    main(args)