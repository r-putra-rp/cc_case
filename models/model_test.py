import sys
import joblib
import json
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model, Sequential

sys.path.append("..")

from utils.logger import get_logger
from utils.config import (
    ASSETS,
    FEAT_DATA_FOLDER,
    FEAT_DATA_FILE_VALIDATION,
    FEAT_DATA_FILE_TRAIN,
    FEAT_SCALER_FILE_TRAIN,
    MODEL_FOLDER,
    MODEL_FILE,
    PRECISION,
    RESULTS_FOLDER,
    POSITION_HISTORIES,
    DECISION_HISTORIES,
)
from model_train import (
    SEQUENCE,
)


logger = get_logger()

ORDER_QTY = 1000
COLUMN_SCALING = [
    "open_",
    "close_",
    "volume_",
    "daily_pct_change_",
    "daily_close_",
    "daily_open_",
]
FEATURES = ["scaled_volume_", "scaled_open_", "hours_to_close"]
FEATURES_ORI = ["volume_", "open_", "hours_to_close"]

class ModelTest:
    def __init__(self) -> None:
        self.dfs_test: Dict[str, pd.DataFrame] = {}
        self.dfs_train: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Sequential] = {}
        self.scalers: Dict[str, RobustScaler] = {}
        self.assets = [x.split(".")[0].lower() for x in ASSETS]
        self.position_histories: Dict[str, List[dict]] = {}
        self.decisions: Dict[str, int] = {}

    def main(self):
        self.load_data()

        for asset in self.assets:
            df_test = self.dfs_test[asset]
            df_train = self.dfs_train[asset]
            scaler = self.scalers[asset]
            model = self.models[asset]

            position_history, decision = self.proces_df(
                asset, df_test, df_train, scaler, model
            )
            self.position_histories[asset] = position_history
            self.decisions[asset] = decision

        with open(f"{RESULTS_FOLDER}/{POSITION_HISTORIES}", "w") as fp:
            json.dump(self.position_histories, fp)

        logger.debug(self.decisions)
        with open(f"{RESULTS_FOLDER}/{DECISION_HISTORIES}", "w") as fp:
            json.dump(self.decisions, fp)

    @staticmethod
    def proces_df(
        asset: str,
        df_test: pd.DataFrame,
        df_train: pd.DataFrame,
        scaler: RobustScaler,
        model: Sequential,
    ) -> Tuple[List[dict], List[int]]:

        features = [f"{x}{asset}" if x != "hours_to_close" else x for x in FEATURES]
        features_ori = [
            f"{x}{asset}" if x != "hours_to_close" else x for x in FEATURES_ORI
        ]
        position_open: dict = None
        position_hist: List[dict] = []
        decision_hist: List[int] = []
        logger.debug(f"df_train \n{df_train.tail()}")
        logger.debug(f"df_test \n{df_test.head()}")

        train_sequence = df_train[features].tail(SEQUENCE).values
        scaler.fit(train_sequence)
        train_sequence = scaler.transform(train_sequence)
        cnt = 0
        pred = 0
        for idx, row_ori in df_test.iterrows():

            logger.debug(f" === Processing {asset} index {idx} === ")
            if position_open is not None:
                if row_ori["hours_to_close"] == 1:
                    open_price = position_open["open_price"]
                    qty = position_open["qty"]
                    close_price = row_ori[f"close_{asset}"]
                    close_ts = row_ori["timestamp"]

                    pnl = (close_price - open_price) / open_price
                    pnl = round(pnl * qty, PRECISION)
                    position_hist.append(
                        {
                            "open_price": open_price,
                            "qty": qty,
                            "close_price": close_price,
                            "pnl": pnl,
                            "close_ts": close_ts.isoformat()
                        }
                    )
                    position_open = None

            row = row_ori[features_ori]
            test_row = row.values.reshape(1, -1)
            test_row = scaler.transform(test_row)

            combined_sequence_scaled = np.vstack((train_sequence, test_row))
            combined_sequence_scaled = combined_sequence_scaled[-SEQUENCE:].reshape(
                (1, SEQUENCE, len(features_ori))
            )

            prediction = model.predict(combined_sequence_scaled)[0][0]
            decision = (prediction > 0.8).astype(int)
            if decision and position_open is None:
                cnt += 1
                position_open = {
                    "open_price": row_ori[f"close_{asset}"],
                    "qty": ORDER_QTY,
                    "close_price": None,
                    "pnl": None,
                }

            pred += 1
            logger.debug(f"Prediction {prediction}")
            logger.debug(f"Decision {decision}")
            logger.debug(f"Count OK {cnt}")
            logger.debug(f"Count Rows {pred}")

            decision_hist.append(int(decision) if decision else 0)

            train_sequence = np.vstack((train_sequence[1:], test_row))

        final_pnl = round(sum(x["pnl"] for x in position_hist), PRECISION)
        logger.debug(f"PNL {final_pnl}")

        return (position_hist, decision_hist)

    def load_data(self):
        for asset in self.assets:

            filename = f"{FEAT_DATA_FOLDER}/{asset}{FEAT_DATA_FILE_TRAIN}"
            logger.debug(f"Reading {asset} filename {filename}")
            df = pd.read_csv(filename, index_col=0)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self.dfs_train[asset] = df

            filename = f"{FEAT_DATA_FOLDER}/{asset}{FEAT_DATA_FILE_VALIDATION}"
            logger.debug(f"Reading {asset} filename {filename}")
            df = pd.read_csv(filename, index_col=0)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self.dfs_test[asset] = df

            filename = f"{FEAT_DATA_FOLDER}/{asset}{FEAT_SCALER_FILE_TRAIN}"
            logger.debug(f"Reading {asset} filename {filename}")
            scaler = joblib.load(filename)
            self.scalers[asset] = scaler

            filename = f"{MODEL_FOLDER}/{asset}{MODEL_FILE}"
            logger.debug(f"Reading {asset} filename {filename}")
            model = load_model(filename)
            self.models[asset] = model


if __name__ == "__main__":
    model_test = ModelTest()
    model_test.main()
