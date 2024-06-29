import sys
import joblib
import json
import pandas as pd
from typing import Dict
from sklearn.preprocessing import RobustScaler
from numpy import cov
from scipy.stats import spearmanr, pearsonr, ttest_ind, mannwhitneyu

sys.path.append("..")

from utils.logger import get_logger
from utils.config import (
    ASSETS,
    PRECISION,
    FEAT_DATA_FOLDER,
    FEAT_DATA_FILE_TRAIN,
    FEAT_SCALER_FILE_TRAIN,
    STD_TRESHOLD,
)

logger = get_logger()


class StatisticalTesting:
    def __init__(self) -> None:
        self.dfs_train: Dict[str, pd.DataFrame] = {}
        self.scalers: Dict[str, RobustScaler] = {}

        self.corrs: Dict[str, float] = {}
        self.ttests: Dict[str, float] = {}

    def main(self):
        self.load_data()
        self.check_spikes()

        self.correlations()
        self.test_statistics()

        logger.debug(
            json.dumps(
                dict(
                    sorted(self.corrs.items(), key=lambda item: item[1], reverse=True)
                ),
                indent=4,
            )
        )
        logger.debug(
            json.dumps(
                dict(
                    sorted(self.ttests.items(), key=lambda item: item[1], reverse=False)
                ),
                indent=4,
            )
        )

    def test_statistics(self):
        for asset, df in self.dfs_train.items():
            df_spike = df[df["is_volume_spike"] == 1]
            df_no_spike = df[df["is_volume_spike"] == 0]
            logger.debug(
                f"{asset} Length of spikes {len(df_spike.index)} no spikes {len(df_no_spike.index)}"
            )
            _, p_value_ttest = ttest_ind(
                df_spike[f"scaled_daily_pct_change_{asset}"].tolist(),
                df_no_spike[f"scaled_daily_pct_change_{asset}"].to_list(),
                equal_var=False,
            )
            _, p_value_mannwhitney = mannwhitneyu(
                df_spike[f"scaled_daily_pct_change_{asset}"].tolist(),
                df_no_spike[f"scaled_daily_pct_change_{asset}"].tolist(),
                alternative="two-sided",
            )

            logger.debug(f"{asset} T-test p-value: {p_value_ttest}")
            logger.debug(f"{asset} Mann-Whitney U test p-value: {p_value_mannwhitney}")

            self.ttests[asset] = round(p_value_ttest, PRECISION)

    def correlations(self):
        for asset, df in self.dfs_train.items():
            df_spike = df[df["is_volume_spike"] == 1]

            logger.debug("\n\n\n\n")
            simple_corr = df_spike[
                [f"scaled_volume_{asset}", f"scaled_daily_pct_change_{asset}"]
            ].corr()
            logger.debug(f"{asset} simple correlation {simple_corr}")

            covariance = cov(
                df_spike[f"scaled_volume_{asset}"].tolist(),
                df_spike[f"scaled_daily_pct_change_{asset}"].tolist(),
            )
            logger.debug(f"{asset} covariance \n{covariance}")

            pearson_corr = pearsonr(
                df_spike[f"scaled_volume_{asset}"].tolist(),
                df_spike[f"scaled_daily_pct_change_{asset}"].tolist(),
            )
            logger.debug(f"{asset} pearson correlation {pearson_corr}")

            spearman_corr = spearmanr(
                df_spike[f"scaled_volume_{asset}"].tolist(),
                df_spike[f"scaled_daily_pct_change_{asset}"].tolist(),
            )
            logger.debug(f"{asset} spearman correlation {spearman_corr}")

            self.corrs[asset] = round(pearson_corr.statistic, PRECISION)

    def check_spikes(self):
        for asset, df in self.dfs_train.items():
            logger.debug(f"Checking spikes in {asset}")

            df["is_volume_spike"] = [
                0 if x < (STD_TRESHOLD * y) else 1
                for x, y in zip(
                    df[f"scaled_volume_{asset}"], df[f"scaled_volume_{asset}_stdev"]
                )
            ]
            self.dfs_train[asset] = df

            logger.debug(f"Number of spikes {sum(df['is_volume_spike'])}")

    def load_data(self):
        for asset in ASSETS:
            asset_name = asset.split(".")[0].lower()
            filename = f"{FEAT_DATA_FOLDER}/{asset_name}{FEAT_DATA_FILE_TRAIN}"
            logger.debug(f"Reading {asset_name} filename {filename}")

            df = pd.read_csv(filename, index_col=0)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self.dfs_train[asset_name] = df

            filename = f"{FEAT_DATA_FOLDER}/{asset_name}{FEAT_SCALER_FILE_TRAIN}"
            logger.debug(f"Reading {asset} scalers {filename}")
            scaler = joblib.load(filename)
            self.scalers[asset_name] = scaler

            logger.debug(f"{asset_name} dataframe head \n{df.head()}")
            logger.debug(f"{asset_name} dataframe tail \n{df.tail()}")
            logger.debug(f"{asset_name} dataframe desc \n{df.describe()}")
            logger.debug(f"{asset_name} dataframe types \n{df.dtypes}")
            logger.debug(f"{asset_name} dataframe info \n{df.info()}")


if __name__ == "__main__":
    stat_test = StatisticalTesting()
    stat_test.main()
