"""
Data processor.
Author: JiaWei Jiang

This file contains the definition of data processor cleaning and
processing raw data before entering modeling phase. Because data
processing is case-specific, I leave flexibility for users to
customize the procedure themselves.
"""
import os
import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from metadata import TARGET
from paths import DUMP_PATH
from validation.holdout import HoldoutSplitter

from .fe import FE


class DataProcessor:
    """Data processor processing raw data and providing access to the
    processed data ready to be fed into models.

    Parameters:
       file_path: path of the raw data
           *Note: File reading supports .parquet extension in default
               setting, which can be modified to customized one.
       dp_cfg: hyperparameters of data processor
    """

    fe: FE
    holdout_splitter: Optional[HoldoutSplitter] = None
    _X: Union[pd.DataFrame, np.ndarray]
    _y: Union[pd.DataFrame, np.ndarray]

    def __init__(self, file_path: str, **dp_cfg: Any):
        if file_path.endswith("xlsx"):
            self._df = pd.read_excel(file_path)
        elif file_path.endswith("csv"):
            self._df = pd.read_csv(file_path)
        self._dp_cfg = dp_cfg
        self._setup()

    def run_before_cv(self) -> None:
        """Clean and process data before cross validation process.

        Holdout set splitting is also done in this process if holdout
        strategy is specified.

        Return:
            None
        """
        print("Run data cleaning and processing before data splitting...")

        # Handle missing values
        self._df.replace(0, np.nan, inplace=True)
        if self.imp_nan:
            self._df.fillna(0, inplace=True)

        # Handle duplicated samples
        if self.drop_dup_samples:
            self._df.drop_duplicates(inplace=True, ignore_index=True)
        if self.stats_to_shrink_dup_feats is not None:
            self._shrink_dup_feats(self.stats_to_shrink_dup_feats)

        # Run feature engineering
        self._run_fe()

        # Split datasets and holdout
        self._split_X_y()
        self._holdout()

    def run_after_splitting(
        self,
        X_tr: Union[pd.DataFrame, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        fold: int,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], object]:
        """Clean and process data after data splitting.

        To avoid data leakage, some data processing techniques should
        be applied after data is splitted.

        Parameters:
            X_tr: X training set
            X_val: X validation set
            fold: current fold number

        Return:
            X_tr: processed X training set
            X_val: processed X validation set
            scl: fittet scaler
        """
        print("Run data cleaning and processing after data splitting...")
        scl = None
        if self.scale_cfg["type"] is not None:
            X_tr, X_val, scl = self._scale(X_tr, X_val)
            # =TODO=
            # Dump trafos and other objects elsewhere
            with open(os.path.join(DUMP_PATH, "trafos", f"fold{fold}.pkl"), "wb") as f:
                pickle.dump(scl, f)

        return X_tr, X_val, scl

    def get_df(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return raw or processed DataFrame (i.e., current snapshot)."""
        return self._df

    def get_X_y(
        self,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        """Return X set and y set."""
        return self._X, self._y

    def get_cat_feats(self) -> List[str]:
        """Return list of categorical features."""
        return self.fe.get_cat_feats()

    def _setup(self) -> None:
        """Retrieve all parameters used in data processing pipeline and
        setup feature engineer.
        """
        # Specify process mode
        if self._dp_cfg.get("infer") is not None:
            self._infer = True
        else:
            self._infer = False

        # Configure raw features and setup feature engineer
        self.feats = self._dp_cfg["feats"]
        self.fe_cfg = self._dp_cfg["fe"]
        self.fe_cfg["infer"] = self._infer
        self.fe = FE(**self.fe_cfg)

        # Before data splitting
        self.imp_nan = self._dp_cfg["imp_nan"]
        self.drop_dup_samples = self._dp_cfg["drop_dup_samples"]
        self.stats_to_shrink_dup_feats = self._dp_cfg["stats_to_shrink_dup_feats"]
        self.holdout_cfg = self._dp_cfg["holdout"]

        # After data splitting
        self.scale_cfg = self._dp_cfg["scale"]

    def _shrink_dup_feats(self, y_stats: str) -> None:
        """Shrink samples with the same feature vector but different
        targets.

        Parameters:
            y_stats: stats used to aggregate different target vectors

        Return:
            None
        """
        if y_stats == "mean":
            df_y_stats = self._df.groupby(self.feats)[TARGET].mean()
        elif y_stats == "median":
            df_y_stats = self._df.groupby(self.feats)[TARGET].median()
        self._df = self._df.merge(df_y_stats, how="left", on=self.feats, suffixes=("", "_"))
        self._df.drop(TARGET, axis=1, inplace=True)
        self._df.rename({f"{target}_": f"{target}" for target in TARGET}, axis=1, inplace=True)
        self._df.drop_duplicates(inplace=True, ignore_index=True)

    def _run_fe(self) -> None:
        """Run feature engineering."""
        print("Start feature engineering...")
        self._df = self.fe.run(self._df)
        print("Done.")

    def _split_X_y(self) -> None:
        """Split data into X and y sets."""
        feats = self.feats  # Features in the unprocessed DataFrame

        # Add newly engineered features into feature set
        # Note: The ordering of DataFrame columns matters
        for ft in self.fe.get_eng_feats():
            if ft not in feats:
                feats.append(ft)

        print("Start splitting X and y set...")
        print(f"Feature set:\n{feats}")
        self._X = self._df[feats]
        self._y = self._df["sensor_point5_i_value"]  # Hard-coded tmp.
        print("Done.")

    def _holdout(self) -> None:
        """Setup holdout splitter, and split the holdout sets."""
        holdout_n_splits = self.holdout_cfg["n_splits"]
        if holdout_n_splits == 0:
            print("Holdout set splitting is disabled, so no local unseen " "test set is used in evaluation process.")
        else:
            self.holdout_splitter = HoldoutSplitter(**self.holdout_cfg)

            print("Start splitting holdout sets for final evaluation...")
            self.holdout_splitter.split(self._X)
            print("Done.")

    # After data splitting
    def _scale(
        self,
        X_tr: Union[pd.DataFrame, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], Any]:
        """Scale numeric features.

        Support only pd.DataFrame now.

        Return:
            X_tr: scaled X training set
            X_val: scaled X validation set
            scl: fittet scaler
        """
        assert isinstance(X_tr, pd.DataFrame) and isinstance(X_val, pd.DataFrame)

        scl_type = self.scale_cfg["type"]
        cols_to_trafo = self.scale_cfg["cols"]

        if scl_type == "minmax":
            scl = MinMaxScaler()
        elif scl_type == "standard":
            scl = StandardScaler()
        elif scl_type == "quantile":
            n_quantiles = self.scale_cfg["n_quantiles"]
            scl = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=168,
            )

        if cols_to_trafo == []:
            cols_to_trafo = _get_numeric_cols(X_tr)

        print(f"Start scaling features using {scl_type} trafo...\n" f"Feature list:\n{cols_to_trafo}")
        X_tr[cols_to_trafo] = scl.fit_transform(X_tr[cols_to_trafo])
        X_val[cols_to_trafo] = scl.transform(X_val[cols_to_trafo])
        print("Done.")

        X_tr.fillna(0, inplace=True)
        X_val.fillna(0, inplace=True)

        return X_tr, X_val, scl


def _get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric column names.

    Parameters:
        df: DataFrame to process

    Return:
        numeric_cols: list of numeric column names.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_cols
