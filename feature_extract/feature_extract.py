import sys
import joblib
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import RobustScaler

sys.path.append("..")

from utils.logger import get_logger
from utils.config import (
    ASSETS,
    ROLLING_WINDOW,
    DATA_FOLDER,
    DATA_FILE,
    PRECISION,
    TRAIN_TEST_SPLIT,
    FEAT_DATA_FOLDER,
    FEAT_DATA_FILE_TRAIN,
    FEAT_DATA_FILE_TEST,
    FEAT_SCALER_FILE_TRAIN,
)

logger = get_logger()

MARKET_CLOSE_TIME = 16
ASSETS = ["BBCA.JK"]

class FeatureExtract:
    def __init__(self):
        self.dfs: Dict[str, pd.DataFrame] = {}
        self.dfs_train: Dict[str, pd.DataFrame] = {}
        self.dfs_test: Dict[str, pd.DataFrame] = {}
        self.scalers_train: Dict[str, RobustScaler] = {}
        self.cols: Dict[str, List[str]] = {}
        self.scaled_cols: Dict[str, List[str]] = {}

    def main(self):
        self.load_data()
        self.process_features()
        self.save_to_csv()

    def process_features(self):
        for asset, df in self.dfs.items():
            df = self.process_time(df)

            # Splitting train-test
            logger.debug(f"Splitting {asset}")
            df_train, df_test = self.split_train_test(df)

            logger.debug(f"Calculating %change daily {asset}")
            df_train = self.process_daily_pct(df_train)
            logger.debug(f"{asset} dataframe tail \n{df_train.tail(20)}")
            exit()
            logger.debug(f"Scaling {asset}")
            df_train = self.scale_data(df_train, asset)

            logger.debug(f"Calculating standard deviation for {asset}")
            df_train = self.process_stdev(df_train, asset)

            logger.debug(f"{asset} dataframe tail \n{df_train.tail()}")

            self.dfs_train[asset] = df_train
            self.dfs_test[asset] = df_test

    @staticmethod
    def process_daily_pct(
        df:pd.DataFrame
    ) -> pd.DataFrame:
        
        daily_pct_change = df.groupby("date").apply(
            lambda x: ((x["close_bbca"].iloc[-1] - x["open_bbca"].iloc[0]) / x["open_bbca"].iloc[0])
        ).reset_index(name="daily_pct_change")

        df = pd.merge(df, daily_pct_change, on="date")
        return df

    def process_stdev(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        for col in self.scaled_cols[asset]:
            df[f"{col}_stdev"] = (
                df[col].rolling(window=ROLLING_WINDOW).std().round(PRECISION)
            )
        return df

    def scale_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        cols = self.cols[asset]
        scaler = RobustScaler()
        scaled_df = scaler.fit_transform(df[cols])

        scaled_df = pd.DataFrame(scaled_df, columns=cols)

        new_cols = {col: f"scaled_{col}" for col in cols}

        scaled_df = scaled_df.rename(columns=new_cols)

        self.scalers_train[asset] = scaler
        self.scaled_cols[asset] = list(new_cols.values())

        df = pd.concat([df, scaled_df], axis=1, join="inner")

        return df

    @staticmethod
    def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        len_index = len(df.index)
        train_index = int(len_index * TRAIN_TEST_SPLIT)
        df_train = df.iloc[:train_index]
        df_test = df.iloc[train_index:]

        return df_train, df_test

    @staticmethod
    def process_time(df: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Type {df['timestamp']}")
        df["day"] = df["timestamp"].dt.day
        df["date"] = df["timestamp"].dt.date
        df["hours_to_close"] = MARKET_CLOSE_TIME - df["timestamp"].dt.hour
        return df

    def load_data(self):
        for asset in ASSETS:
            asset_name = asset.split(".")[0].lower()
            filename = f"{DATA_FOLDER}/{asset_name}{DATA_FILE}"
            logger.debug(f"Reading {asset_name} filename {filename}")

            df = pd.read_csv(filename, index_col=0)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self.dfs[asset_name] = df
            cols = list(df.columns)
            cols.remove("timestamp")
            self.cols[asset_name] = cols

            logger.debug(f"{asset_name} dataframe head \n{df.head()}")
            logger.debug(f"{asset_name} dataframe tail \n{df.tail()}")
            logger.debug(f"{asset_name} dataframe desc \n{df.describe()}")
            logger.debug(f"{asset_name} dataframe types \n{df.dtypes}")
            logger.debug(f"{asset_name} dataframe info \n{df.info()}")
            logger.debug(f"{asset_name} columns info \n{self.cols[asset_name]}")
            

    def save_to_csv(self):
        for asset, df in self.dfs_train.items():
            filename = f"{FEAT_DATA_FOLDER}/{asset}{FEAT_DATA_FILE_TRAIN}"
            logger.debug(f"Saving {asset} to {filename}")
            df.to_csv(filename)

        for asset, df in self.dfs_test.items():
            filename = f"{FEAT_DATA_FOLDER}/{asset}{FEAT_DATA_FILE_TEST}"
            logger.debug(f"Saving {asset} to {filename}")
            df.to_csv(filename)

        for asset, scaler in self.scalers_train.items():
            filename = f"{FEAT_DATA_FOLDER}/{asset}{FEAT_SCALER_FILE_TRAIN}"
            logger.debug(f"Saving {asset} scalers to {filename}")
            joblib.dump(scaler, filename)


if __name__ == "__main__":
    feature_extract = FeatureExtract()
    feature_extract.main()
