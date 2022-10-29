"""
Experiment logger.
Author: JiaWei Jiang

This file contains the definition of experiment logger for experiment
configuration, message logging, object dumping, etc.
"""
from __future__ import annotations

import os
import pickle
from argparse import Namespace
from shutil import rmtree  # , make_archive
from types import TracebackType
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator

import wandb
from config.config import gen_exp_id, setup_dp, setup_model, setup_proc
from paths import DUMP_PATH, OOF_META_FEATS_PATH


class Experiment(object):
    """Experiment logger.

    Parameters:
        args: arguments driving training and evaluation processes
        dl: if the experiment uses DL models, then set to True
    """

    cfg: Dict[str, Dict[str, Any]]
    model_params: Dict[str, Any]
    fit_params: Optional[Dict[str, Any]] = {}

    def __init__(self, args: Namespace, dl: bool = False):
        self.exp_id = gen_exp_id(args.model_name)
        self.args = args
        self.dp_cfg = setup_dp()
        self.model_cfg = setup_model(args.model_name)
        self.dl = dl
        if dl:
            self.proc_cfg = setup_proc()

        # Post process, parse and aggregate configuration
        self._parse_model_cfg()
        self._agg_cfg()

        self._mkbuf()

    def __enter__(self) -> Experiment:
        self._run()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_inst: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._halt()

    def dump_cfg(self, cfg: Dict[str, Any], file_name: str) -> None:
        """Dump config dictionary to corresponding path.

        Parameters:
            cfg: configuration
            file_name: name of the file with .yaml extension

        Return:
            None
        """
        dump_path = os.path.join(DUMP_PATH, "config", f"{file_name}.yaml")
        with open(dump_path, "w") as f:
            yaml.dump(cfg, f)

    def dump_ndarr(self, arr: np.ndarray, file_name: str) -> None:
        """Dump np.ndarray under corresponding path.

        Parameters:
            arr: array to dump
            file_name: name of the file with .npy extension

        Return:
            None
        """
        if file_name.startswith("oof"):
            dump_path = os.path.join(DUMP_PATH, "preds", "oof", file_name)
        elif file_name.startswith("holdout"):
            dump_path = os.path.join(DUMP_PATH, "preds", "holdout", file_name)
        np.save(dump_path, arr)

    def dump_df(self, df: pd.DataFrame, file_name: str, ext: str = "parquet") -> None:
        """Dump DataFrame under corresponding path.

        Support only for dumping feature importance df now.

        Parameters:
            file_name: name of the file with . extension
        """
        dump_path = os.path.join(DUMP_PATH, file_name)
        df.to_parquet(f"{dump_path}.{ext}", index=False)

    def dump_model(self, model: BaseEstimator, model_type: str, mid: int) -> None:
        """Dump estimator to corresponding path.

        Parameters:
            model: well-trained estimator
            model_type: type of the model, the choices are as follows:
                {"fold", "whole"}
            mid: identifer of the model

        Return:
            None
        """
        model_file_prefix = "fold" if model_type == "fold" else "seed"
        dump_path = os.path.join(DUMP_PATH, "models", model_type, f"{model_file_prefix}{mid}.pkl")
        with open(dump_path, "wb") as f:
            pickle.dump(model, f)

    def incorp_meta_feats(self, pred: np.ndarray) -> None:
        """Incorporate the predicting results into meta feature pool.

        Parameters:
            pred: predicting results

        Return:
            None
        """
        if not os.path.exists(OOF_META_FEATS_PATH):
            meta_feats = pd.DataFrame()
        else:
            meta_feats = pd.read_csv(OOF_META_FEATS_PATH)

        if self.dp_cfg["fe"]["meta_feats"] != []:
            # Stacking or restacking is triggered
            # Tmp workaround, it's better to used merge to avoid unalignment
            pred_padded = np.zeros(len(meta_feats))
            pred_padded[-len(pred) :] = pred
            pred = pred_padded
        meta_feats[self.exp_id] = pred
        meta_feats.to_csv(OOF_META_FEATS_PATH, index=False)

    def _parse_model_cfg(self) -> None:
        """Configure model parameters and parameters passed to `fit`
        method if they're provided.
        """
        if self.dl:
            self.model_params = self.model_cfg
        else:
            self.model_params = self.model_cfg["model_params"]
            if self.model_cfg["fit_params"] is not None:
                self.fit_params = self.model_cfg["fit_params"]

    def _agg_cfg(self) -> None:
        """Aggregate sub configurations of different components into
        one summarized configuration.
        """
        self.cfg = {
            "common": vars(self.args),
            "dp": self.dp_cfg,
            "model": self.model_params,
            "fit": self.fit_params,
        }
        if self.dl:
            self.cfg["proc"] = self.proc_cfg

    def _mkbuf(self) -> None:
        """Make local buffer for experiment output dumping."""
        if os.path.exists(DUMP_PATH):
            rmtree(DUMP_PATH)
        os.mkdir(DUMP_PATH)
        os.mkdir(os.path.join(DUMP_PATH, "config"))
        os.mkdir(os.path.join(DUMP_PATH, "models"))
        for model_type in ["fold", "whole"]:
            os.mkdir(os.path.join(DUMP_PATH, "models", model_type))
        os.mkdir(os.path.join(DUMP_PATH, "trafos"))
        os.mkdir(os.path.join(DUMP_PATH, "preds"))
        for pred_type in ["oof", "holdout"]:
            os.mkdir(os.path.join(DUMP_PATH, "preds", pred_type))
        os.mkdir(os.path.join(DUMP_PATH, "imp"))

    def _run(self) -> None:
        """Start a new experiment entry."""
        self.exp_supr = wandb.init(
            project=self.args.project_name,
            config=self.cfg,
            group=self.exp_id,
            job_type="supervise",
            name="supr",
        )
        self._log_exp_metadata()
        self.exp_supr.finish()

    def _log_exp_metadata(self) -> None:
        """Log metadata of the experiment to Wandb."""
        print(f"=====Experiment {self.exp_id}=====")
        print(f"-> Model: {self.args.model_name}")
        print(f"-> CV Scheme: {self.args.cv_scheme}")
        print(f"-> Holdout Strategy: {self.dp_cfg['holdout']}")

    def _halt(self) -> None:
        # Push artifacts to remote
        dump_entry = wandb.init(project=self.args.project_name, group=self.exp_id, job_type="dumping")
        artif = wandb.Artifact(name=self.args.model_name.upper(), type="output")
        artif.add_dir(DUMP_PATH)
        dump_entry.log_artifact(artif)
        dump_entry.finish()

        # Compress local outputs
        # make_archive(f"./{self.exp_id}", "zip", root_dir="./", base_dir=DUMP_PATH)
