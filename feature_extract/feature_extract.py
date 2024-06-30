import sys
import joblib
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import RobustScaler

sys.path.append("..")

from utils.logger import get_logger
from utils.config import (
    ASSETS,
    ROLLING_WINDOW_HR,
    ROLLING_WINDOW_DAY,
    DATA_FOLDER,
    DATA_FILE,
    PRECISION,
    FEAT_DATA_FOLDER,
    FEAT_DATA_FILE_TRAIN,
    FEAT_DATA_FILE_VALIDATION,
    FEAT_SCALER_FILE_TRAIN,
)

logger = get_logger()

MARKET_CLOSE_TIME = 16
VALIDATION_SPLIT = 0.9


class FeatureExtract:
    def __init__(self):
        self.dfs: Dict[str, pd.DataFrame] = {}
        self.dfs_train: Dict[str, pd.DataFrame] = {}
        self.dfs_validation: Dict[str, pd.DataFrame] = {}
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
            df_train, df_validation = self.split_train_validation(df)

            logger.debug(f"Calculating %change and close daily {asset}")
            df_train = self.process_daily_data(df_train, asset)

            logger.debug(f"Scaling {asset}")
            df_train = self.scale_data(df_train, asset)

            logger.debug(f"Calculating standard deviation for {asset}")
            df_train = self.process_stdev(df_train, asset)

            logger.debug(f"{asset} dataframe tail \n{df_train.tail()}")

            self.dfs_train[asset] = df_train
            self.dfs_validation[asset] = df_validation

    def process_daily_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        name_pct = f"daily_pct_change_{asset}"
        daily_pct_change = (
            df.groupby("date")
            .apply(
                lambda x: (
                    (x[f"close_{asset}"].iloc[-1] - x[f"open_{asset}"].iloc[0])
                    / x[f"open_{asset}"].iloc[0]
                )
            )
            .reset_index(name=name_pct)
        )
        df = pd.merge(df, daily_pct_change, on="date")

        name_open = f"daily_open_{asset}"
        daily_open = (
            df.groupby("date")
            .apply(
                lambda x: (
                    x[f"open_{asset}"].iloc[0] 
                )
            )
            .reset_index(name=name_open)
        )
        df = pd.merge(df, daily_open, on="date")

        name_close = f"daily_close_{asset}"
        daily_close = (
            df.groupby("date")
            .apply(
                lambda x: (
                    x[f"close_{asset}"].iloc[-1] 
                )
            )
            .reset_index(name=name_close)
        )
        df = pd.merge(df, daily_close, on="date")

        cols = self.cols[asset]
        cols.extend([name_pct, name_close, name_open])
        self.cols[asset] = cols

        return df

    def process_stdev(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:

        daily_pct_change_col = f"scaled_daily_pct_change_{asset}"

        pct_chg = df[["date", daily_pct_change_col]].copy(deep=True).drop_duplicates()
        pct_chg[f"daily_pct_change_{asset}_stdev"] = (
            df[daily_pct_change_col]
            .rolling(window=ROLLING_WINDOW_DAY)
            .std()
            .round(PRECISION)
        )

        for col in self.scaled_cols[asset]:
            if col != daily_pct_change_col:
                df[f"{col}_stdev"] = (
                    df[col].rolling(window=ROLLING_WINDOW_HR).std().round(PRECISION)
                )

        df = pd.merge(df, pct_chg[["date", f"daily_pct_change_{asset}_stdev"]], on="date")
        

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
    def split_train_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        len_index = len(df.index)
        train_index = int(len_index * VALIDATION_SPLIT)
        df_train = df.iloc[:train_index]
        df_validation = df.iloc[train_index:]

        return df_train, df_validation

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

        for asset, df in self.dfs_validation.items():
            filename = f"{FEAT_DATA_FOLDER}/{asset}{FEAT_DATA_FILE_VALIDATION}"
            logger.debug(f"Saving {asset} to {filename}")
            df.to_csv(filename)

        for asset, scaler in self.scalers_train.items():
            filename = f"{FEAT_DATA_FOLDER}/{asset}{FEAT_SCALER_FILE_TRAIN}"
            logger.debug(f"Saving {asset} scalers to {filename}")
            joblib.dump(scaler, filename)


if __name__ == "__main__":
    feature_extract = FeatureExtract()
    feature_extract.main()
