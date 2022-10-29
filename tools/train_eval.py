"""
Main script for training and evaluation processes using traditional ML.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme. Furthermore, evaluation can be done on unseen
(test) data optionally.
"""
import warnings
from argparse import Namespace

import numpy as np

from cv.build import build_cv
from data.data_processor import DataProcessor
from engine.defaults import TrainEvalArgParser
from experiment.experiment import Experiment
from modeling.build import build_models
from validation.cross_validate import cross_validate

warnings.simplefilter("ignore")


# Define evaluation metric, RMSE
# =TODO=
# Defined separately and pass an `Evaluator` obj to `cross_validate`
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def main(args: Namespace) -> None:
    """Run training and evaluation.

    Parameters:
        args: arguments driving training and evaluation processes

    Return:
        None
    """
    # Configure experiment
    experiment = Experiment(args)

    with experiment as exp:
        # Clean and process data to feed into the model
        exp.dump_cfg(exp.dp_cfg, "dp")
        dp = DataProcessor(args.input_path, **exp.dp_cfg)
        dp.run_before_cv()

        # Build cross validator
        cv = build_cv(args)

        # Build models
        exp.dump_cfg(exp.model_cfg, args.model_name)
        models = build_models(args.model_name, exp.model_params, cv.get_n_splits())

        # Start cv process
        cv_result = cross_validate(
            exp=exp,
            dp=dp,
            models=models,
            cv=cv,
            fit_params=exp.fit_params,
            eval_fn=rmse,
            stratified=args.stratified,
            group=args.group,
        )

        # Dump cv results
        exp.dump_ndarr(cv_result.oof_pred, "oof")
        if cv_result.holdout_pred is not None:
            for fold, holdout in enumerate(cv_result.holdout_pred):
                exp.dump_ndarr(holdout, f"holdout_fold{fold}")

        for fold, model in enumerate(models):
            exp.dump_model(model, model_type="fold", mid=fold)

        for fold, imp in enumerate(cv_result.imp):
            exp.dump_df(imp, f"imp/fold{fold}", ext="parquet")


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
