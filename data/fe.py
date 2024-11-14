"""
Feature engineer.
"""
from typing import List

import pandas as pd

# from paths import OOF_META_FEATS_PATH, TEST_META_FEATS_PATH


class FE:
    """Feature engineer.

    Parameters:
        ...

        infer: whether the process is in inference mode
    """

    MV2EID = {
        "l0": "lgbm-xxxxxxxx",
    }  # Base model version to corresponding experiment identifier
    EPS: float = 1e-7
    _df: pd.DataFrame = None  # FE is applied to this DataFrame (main obj for data flowing)
    _eng_feats: List[str] = []  # Newly engineered features
    _cat_feats: List[str] = []  # Newly engineered categorical features (how about the original ones?)

    def __init__(
        self,
        #         add_inter_subproc_diff: bool,   # Example acting as placeholder
        infer: bool = False,
    ):
        #         self.add_inter_subproc_diff = add_inter_subproc_diff

        self.infer = infer

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run feature engineering.

        Parameters:
            df: input DataFrame

        Return:
            self._df: DataFrame with engineered features
        """
        self._df = df.copy()

        #         if self.add_inter_subproc_diff:
        #             self._add_inter...()

        return self._df

    def get_eng_feats(self) -> List[str]:
        """Return list of all engineered features."""
        return self._eng_feats

    def get_cat_feats(self) -> List[str]:
        """Return list of categorical features."""
        return self._cat_feats

    """Meta features are used for stacking, so leave it here?
    def _add_meta_feats(self) -> None:
        '''Add meta features for stacking or restacking.'''
        if self.infer:
            # Testing prediction is used
            meta_feats = pd.read_csv(TEST_META_FEATS_PATH)
        else:
            # Unseen prediction is used
            meta_feats = pd.read_csv(OOF_META_FEATS_PATH)

        print("Adding meta features...")
        meta_cols = []
        for model_v in self.meta_feats:
            meta_cols.append(self.MV2EID[model_v])
        meta_feats = meta_feats[PK + meta_cols]

        #         for meta_col in meta_cols:
        #             meta_feats[meta_col] = (meta_feats[meta_col]
        #                                     / meta_feats["Capacity"])

        self._df = self._df.merge(meta_feats, how="left", on=PK, validate="1:1")
        print("Done.")

        self._eng_feats += meta_cols

    def _add_knn_meta_feats(self) -> None:
        '''Add meta features from kNN.

        Illustration of kNN meta column conversion:
            {
                "l5": 2,
                "l6": 3,
                ...
                <model version>: k
            }
            -> {
                "lgbm-hjc3rp0j": 2,
                "lgbm-54or6r30": 3,
                ...
                <experiment identifier>: k
            }
        '''
        if self.infer:
            # Testing prediction is used
            meta_feats = pd.read_csv(TEST_META_FEATS_PATH)
        else:
            # Unseen prediction is used
            meta_feats = pd.read_csv(OOF_META_FEATS_PATH)

        # Load geographic kNN of each generator
        with open("./data/processed/gen_geo_knn.pkl", "rb") as f:
            geo_knn = pickle.load(f)

        print("Adding kNN meta features...")
        for model_v, k in self.knn_meta_feats.items():
            self.knn_meta_feats[self.MV2EID[model_v]] = self.knn_meta_feats.pop(model_v)

        knn_meta_cols = [
            f"{meta_col}_n{i}"
            for meta_col, k in self.knn_meta_feats.items()
            for i in range(k)
        ]
        knn_meta_dict: Dict[str, List[float]] = {col: [] for col in knn_meta_cols}
        for i, r in self._df.iterrows():
            meta_feats_date = meta_feats[meta_feats["Date"] == r["Date"]]
            cap = str(r["Capacity"])

            for meta_col, k in self.knn_meta_feats.items():
                knn = geo_knn[cap][:k]

                for i, cap_ in enumerate(knn):
                    cap_ = float(cap_)
                    df_knn = meta_feats_date[meta_feats_date["Capacity"] == cap_]

                    knn_meta_col = f"{meta_col}_n{i}"
                    if len(df_knn) != 0:
                        knn_meta_dict[knn_meta_col].append(
                            df_knn[meta_col].values[0]  # / cap_
                        )
                    else:
                        knn_meta_dict[knn_meta_col].append(np.nan)

        knn_meta_df = pd.DataFrame.from_dict(knn_meta_dict)
        self._df = pd.concat([self._df, knn_meta_df], axis=1)
        print("Done.")

        self._eng_feats += knn_meta_cols

    """
