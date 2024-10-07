import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, DateOffset


def dasarian_calendar(start, periods=1000):
    start -= DateOffset(months=1)
    dasarian_holidays = [dt.date() for dt in pd.date_range(start, periods=10*(periods+1), freq="D") if dt.day not in [5, 15, 25]]
    return CustomBusinessDay(weekmask='1111111', holidays=dasarian_holidays)
