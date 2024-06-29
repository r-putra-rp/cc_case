import pytz
from datetime import datetime, timedelta

# IDX30 For now
ASSETS = [
    "ACES.JK",
    "ADRO.JK",
    "AKRA.JK",
    "AMRT.JK",
    "ANTM.JK",
    "ARTO.JK",
    "ASII.JK",
    "BBCA.JK",
    "BBNI.JK",
    "BBRI.JK",
    "BMRI.JK",
    "BRPT.JK",
    "BUKA.JK",
    "CPIN.JK",
    "GOTO.JK",
    "ICBP.JK",
    "INCO.JK",
    "INDF.JK",
    "INKP.JK",
    "ITMG.JK",
    "KLBF.JK",
    "MDKA.JK",
    "MEDC.JK",
    "PGAS.JK",
    "PGEO.JK",
    "PTBA.JK",
    "SMGR.JK",
    "TLKM.JK",
    "UNTR.JK",
    "UNVR.JK",
]
ROLLING_WINDOW = 30
GRANULARITY = "1h"

WIB = pytz.timezone('Asia/Jakarta')
DATA_COLLECTION_END = datetime.now().astimezone(WIB)  # Current timestamp
DATA_COLLECTION_START = DATA_COLLECTION_END - timedelta(days=729)  # Depends on GRANURALITY, if its 1hr, max data from yf is 730 days exclusive

DATA_COLLECTION_END_UNIX = int(DATA_COLLECTION_END.timestamp())
DATA_COLLECTION_START_UNIX = int(DATA_COLLECTION_START.timestamp())

DATA_FOLDER = "../ticker_data"
DATA_FILE = "_daily_ticker.csv"  # ie bbca_daily_ticker.csv

PRECISION = 6


class ConstantMultipliers:
    NANO = 1_000_000_000
    MICRO = 1_000_000
    MILI = 1_000

class SecondsMultipliers:
    SECOND = 1
    MINUTE = 60
    HOUR = 3_600
    DAY = 86_400
    WEEK = 604_800
    MONTH = 2_592_000  # Approx 30 days
    YEAR = 31_536_000  # Approx 365 Days

class HoursMultipliers:
    SECOND = round(1/3_600, PRECISION)
    MINUTE = round(1/3_600, PRECISION)
    HOUR = round(float(1), PRECISION)
    DAY = round(float(24), PRECISION)
    WEEK = round(float(24 * 7), PRECISION)
    MONTH = round(float(24 * 30), PRECISION)  # Approx 30 days
    YEAR = round(float(24 * 365), PRECISION)  # Approx 365 Days
