import sys
sys.path.append("..")

import pandas as pd
import requests

from traceback import format_exc
from typing import Dict
from time import sleep

from utils.logger import get_logger
from utils.config import (
    ASSETS,
    DATA_COLLECTION_START,
    DATA_COLLECTION_END,
    DATA_COLLECTION_START_UNIX,
    DATA_COLLECTION_END_UNIX,
    WIB,
    DATA_FOLDER,
    DATA_FILE,
    GRANULARITY,
    PRECISION,
    SecondsMultipliers,
)

logger = get_logger()

CONN_RETRY_QUOTA = 5
TIMEOUT = 2

YAHOO_FINANCE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/"


class DataCollector:
    def __init__(self) -> None:
        self.dfs: Dict[str, pd.DataFrame] = {}

    def main(self):
        self.get_yf_data()
        self.check_data()
        self.save_to_csv()
        sleep(SecondsMultipliers.SECOND)

    def save_to_csv(self):
        for asset, df in self.dfs.items():
            filename = f"{DATA_FOLDER}/{asset}{DATA_FILE}"
            logger.debug(f"Saving {asset} to {filename}")
            df.to_csv(filename)

    def check_data(self):
        # TODO:
        # Check data integrity
        # Assume OK for now
        # for asset, df in self.dfs.items():
        # Do Something
        pass

    def get_yf_data(
        self,
    ):
        for asset in ASSETS:
            logger.info("\n\n")
            logger.info(
                f" == Fetching {asset} from yahoo finance from {DATA_COLLECTION_START}  to {DATA_COLLECTION_END} == "
            )

            asset_name = asset.split(".")[0].lower()

            retry = 0
            while retry <= CONN_RETRY_QUOTA:
                try:
                    url = f"{YAHOO_FINANCE_URL}{asset}?period1={DATA_COLLECTION_START_UNIX}&period2={DATA_COLLECTION_END_UNIX}&interval={GRANULARITY}&includePrePost=False"
                    headers = {
                        "Accept": "*/*",
                        "Accept-Encoding": "gzip, deflate, br, zstd",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Content-Type": "text/plain",
                        "Origin": "https://finance.yahoo.com",
                        "Priority": "u=1, i",
                        "Referer": f"https://finance.yahoo.com/quote/{asset}/",
                        "Sec-Ch-Ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Microsoft Edge";v="126"',
                        "Sec-Ch-Ua-Mobile": "?0",
                        "Sec-Ch-Ua-Platform": "Windows",
                        "Sec-Fetch-Dest": "empty",
                        "Sec-Fetch-Mode": "cors",
                        "Sec-Fetch-Site": "cross-site",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
                    }
                    logger.debug(f"Connecting to {url}")
                    data = requests.get(url=url, headers=headers, timeout=TIMEOUT)
                    logger.debug(data.text)
                    data: dict = data.json()
                    break

                except Exception as e:
                    logger.error(
                        f"Error {e} while fetching binance data, retrying number {retry} from quota {CONN_RETRY_QUOTA}"
                    )
                    logger.error(format_exc())

                    if retry == CONN_RETRY_QUOTA:
                        raise e

                    retry += 1
                    sleep(SecondsMultipliers.SECOND)

            data: dict = data["chart"]["result"][0]
            timestamps: list = data["timestamp"]
            open_prices: list = data["indicators"]["quote"][0]["open"]
            close_prices: list = data["indicators"]["quote"][0]["close"]
            volumes: list = data["indicators"]["quote"][0]["volume"]

            if not (
                len(timestamps) == len(open_prices) == len(close_prices) == len(volumes)
            ):
                logger.error(f"Received data from yfinance for asset {asset_name}")
                logger.error(f"length of timestamp  {len(timestamps)}")
                logger.error(f"length of open       {len(open_prices)}")
                logger.error(f"length of close      {len(close_prices)}")
                logger.error(f"length of volume     {len(volumes)}")

                class DataLengthMismatchError(Exception):
                    pass

                raise DataLengthMismatchError("Lists have different lengths!")

            df = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    f"open_{asset_name}": open_prices,
                    f"close_{asset_name}": close_prices,
                    f"volume_{asset_name}": volumes,
                }
            )
            df["timestamp"] = df["timestamp"] + (
                SecondsMultipliers.HOUR * 7
            )  # Add 7 hour to GMT + 7
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["timestamp"] = df["timestamp"].dt.tz_localize(WIB)
            df[f"open_{asset_name}"] = df[f"open_{asset_name}"].round(PRECISION)
            df[f"close_{asset_name}"] = df[f"close_{asset_name}"].round(PRECISION)
            df[f"volume_{asset_name}"] = df[f"volume_{asset_name}"].round(PRECISION)

            logger.debug(f"{asset_name} dataframe head \n{df.head()}")
            logger.debug(f"{asset_name} dataframe tail \n{df.tail()}")
            logger.debug(f"{asset_name} dataframe desc \n{df.describe()}")
            logger.debug(f"{asset_name} dataframe types \n{df.dtypes}")

            self.dfs[asset_name] = df


if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.main()
