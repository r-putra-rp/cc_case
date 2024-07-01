import sys
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from typing import Dict, List
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
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
    RESULTS_FOLDER,
    POSITION_HISTORIES,
    DECISION_HISTORIES,
)
from models.model_train import SEQUENCE, FEATURES, PCT_CHANGE_TRESHOLD

logger = get_logger()
RESULT_ANALYSIS_FOLDER = "../results/final_results"


class Analysis:
    def __init__(self) -> None:
        self.assets = [x.split(".")[0].lower() for x in ASSETS]
        self.dfs_test: Dict[str, pd.DataFrame] = {}
        self.dfs_train: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Sequential] = {}
        self.scalers: Dict[str, RobustScaler] = {}
        self.positions: Dict[str, List[dict]] = {}
        self.decisions: Dict[str, int] = {}
        self.confusion_test: Dict[str, pd.DataFrame] = {}
        self.confusion_train: Dict[str, pd.DataFrame] = {}

    def main(self):
        self.load_data()

        for asset in self.assets:
            self.train_confusion(asset)
            self.test_confusion(asset)
            self.pnl_analysis(asset)

        total_pnl = {
            asset: sum(entry["pnl"] for entry in entries)
            for asset, entries in self.positions.items()
        }
        sorted_assets = sorted(total_pnl.items(), key=lambda x: x[1], reverse=True)
        for asset, pnl in sorted_assets:
            logger.debug(f"{asset}: {round(pnl , 2):_}")

    def pnl_analysis(self, asset: str):
        positions = self.positions[asset]

        timestamps = [datetime.fromisoformat(x["close_ts"]) for x in positions]
        pnl_values = [x["pnl"] for x in positions]
        pnl_values = [sum(pnl_values[:i+1]) for i in range(len(pnl_values))]

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, pnl_values, marker="o", linestyle="-")

        plt.title(f"{asset.upper()} PnL Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("PnL")
        plt.xticks(rotation=45)

        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{RESULT_ANALYSIS_FOLDER}/{asset}_pnl.png")

    def test_confusion(self, asset: str):
        df = self.dfs_test[asset]
        df = self.prep_classifier(asset, df)

        true_labels = df["target"].values
        predicted_decisions = self.decisions[asset]

        confusion_df = confusion_matrix(true_labels, predicted_decisions)

        index = [f"Actual Negative {asset.upper()}", f"Actual Positive {asset.upper()}"]
        confusion_df = pd.DataFrame(
            confusion_df,
            index=index,
            columns=[
                f"Predicted Negative {asset.upper()}",
                f"Predicted Positive {asset.upper()}",
            ],
        )

        total_samples = confusion_df.values.sum()
        confusion_df = (confusion_df / total_samples) * 100

        labels = {
            "Actual Negative": "True Negatives",
            "Actual Positive": "False Negatives",
            "Predicted Negative": "False Positives",
            "Predicted Positive": "True Positives",
        }
        confusion_df = confusion_df.rename(index=labels, columns=labels)
        logger.debug(f"\n\nTest Confusion Matrix \n{confusion_df}")

        confusion_df.to_csv(f"{RESULT_ANALYSIS_FOLDER}/{asset}_test_confusion.csv")

    def train_confusion(self, asset: str):
        df = self.dfs_train[asset]
        model = self.models[asset]
        features = [f"{x}{asset}" if x != "hours_to_close" else x for x in FEATURES]

        df = self.prep_classifier(asset, df)

        def prepare_sequences(df, seq_length, features):
            sequences = []
            for i in range(len(df) - seq_length + 1):
                seq = df[features].iloc[i : i + seq_length].values
                sequences.append(seq)
            return np.array(sequences)

        sequences = prepare_sequences(df, SEQUENCE, features)

        predictions = model.predict(sequences)
        predicted_labels = (predictions > 0.5).astype(int).reshape(-1)

        true_labels = df["target"].values[SEQUENCE - 1 :]

        confusion_df = confusion_matrix(true_labels, predicted_labels)

        index = [f"Actual Negative {asset.upper()}", f"Actual Positive {asset.upper()}"]
        confusion_df = pd.DataFrame(
            confusion_df,
            index=index,
            columns=[
                f"Predicted Negative {asset.upper()}",
                f"Predicted Positive {asset.upper()}",
            ],
        )

        total_samples = confusion_df.values.sum()
        confusion_df: pd.DataFrame = (confusion_df / total_samples) * 100

        labels = {
            "Actual Negative": "True Negatives",
            "Actual Positive": "False Negatives",
            "Predicted Negative": "False Positives",
            "Predicted Positive": "True Positives",
        }
        confusion_df = confusion_df.rename(index=labels, columns=labels)

        logger.debug(f"\n\nTrain Confusion Matrix \n{confusion_df}")

        confusion_df.to_csv(f"{RESULT_ANALYSIS_FOLDER}/{asset}_train_confusion.csv")

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

            filename = f"{RESULTS_FOLDER}/{POSITION_HISTORIES}"
            logger.debug(f"Reading {asset} filename {filename}")
            with open(filename) as json_data:
                self.positions = json.load(json_data)

            filename = f"{RESULTS_FOLDER}/{DECISION_HISTORIES}"
            logger.debug(f"Reading {asset} filename {filename}")
            with open(filename) as json_data:
                self.decisions = json.load(json_data)

    @staticmethod
    def prep_classifier(asset: str, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"{asset} calculating classifier")
        df["target"] = (df[f"daily_pct_change_{asset}"] > PCT_CHANGE_TRESHOLD).astype(
            int
        )

        return df


if __name__ == "__main__":
    analysis = Analysis()
    analysis.main()
