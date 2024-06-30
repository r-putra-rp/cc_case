import sys
import joblib
import pandas as pd
import numpy as np

from traceback import format_exc
from typing import Dict
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

sys.path.append("..")

from utils.logger import get_logger
from utils.config import (
    ASSETS,
    FEAT_DATA_FOLDER,
    FEAT_DATA_FILE_TRAIN,
    MODEL_FOLDER,
    MODEL_FILE,
)

logger = get_logger()

PCT_CHANGE_TRESHOLD = 0.01
SEQUENCE = 20 * 7  # Look back x hours
FEATURES = ["scaled_volume_", "scaled_open_", "hours_to_close"]
TARGET = "target"
TRAIN_SIZE = 0.8
EPOCH = 16
BATCH_SIZE = 32


class ModelTest:
    def __init__(self) -> None:
        self.dfs_train: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Sequential] = {}

    def main(self):
        self.load_data()
        self.prep_classifier()
        self.lstm_train()
        self.save_to_file()

    def save_to_file(self):
        for asset, model in self.models.items():
            filename = f"{MODEL_FOLDER}/{asset}{MODEL_FILE}"
            logger.debug(f"saving to {filename}")
            model.save(filename)

    def lstm_train(self):
        for asset, df in self.dfs_train.items():
            features = [f"{x}{asset}" if x != "hours_to_close" else x for x in FEATURES]

            X, y = self.create_sequences(df, features, TARGET, SEQUENCE)

            train_size = int(TRAIN_SIZE * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            model = Sequential()
            model.add(
                LSTM(128, return_sequences=True, input_shape=(SEQUENCE, len(features)))
            )
            model.add(Dropout(0.2))
            model.add(LSTM(128, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(128))
            model.add(Dropout(0.2))
            model.add(Dense(16))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            _ = model.fit(
                X_train,
                y_train,
                epochs=EPOCH,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test),
            )

            loss, accuracy = model.evaluate(X_test, y_test)
            logger.debug(f"{asset} test Loss {loss}")
            logger.debug(f"{asset} accuracy {accuracy}")

            self.models[asset] = model

    @staticmethod
    def create_sequences(df, features, target, sequence_length):
        sequences = []
        targets = []

        for i in range(len(df) - sequence_length):
            sequence = df[features].iloc[i : i + sequence_length].values
            target_value = df[target].iloc[i + sequence_length]
            sequences.append(sequence)
            targets.append(target_value)

        return np.array(sequences), np.array(targets)

    def prep_classifier(self):
        for asset, df in self.dfs_train.items():
            logger.debug(f"{asset} calculating classifier")
            df["target"] = (
                df[f"daily_pct_change_{asset}"] > PCT_CHANGE_TRESHOLD
            ).astype(int)

            self.dfs_train[asset] = df

    def load_data(self):
        for asset in ASSETS:
            asset_name = asset.split(".")[0].lower()
            filename = f"{FEAT_DATA_FOLDER}/{asset_name}{FEAT_DATA_FILE_TRAIN}"
            logger.debug(f"Reading {asset_name} filename {filename}")

            df = pd.read_csv(filename, index_col=0)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self.dfs_train[asset_name] = df


if __name__ == "__main__":
    model_test = ModelTest()
    model_test.main()
