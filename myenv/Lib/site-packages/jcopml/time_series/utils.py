import warnings
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_tester(time_series, dropna=True):
    """
    Summarize ADF and KPSS Stationarity Test and show the conclusion.
    
    Please note that you should have at least 20 data to have an accurate estimates of distribution (MacKinnon, 2010).
    I would recommend to have at least 45 data.

    Reference:
    MacKinnon, J. G. (2010). Critical values for cointegration tests: Queen's Economics Department Working Paper No. 1227. Queen's University, Department of Economics.
    """
    if dropna:
        time_series = time_series.dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_adf = _adf_test(time_series)
        df_kpss = _kpss_test(time_series)

    df = pd.concat([df_adf, df_kpss], axis=1, sort=False).T
    _adf, _kpss = df.reject_null.astype(bool)

    if _adf:
        print("ADF:        Series is stationary")
    else:
        print("ADF:        Series is non-stationary")

    if _kpss:
        print("KPSS:       Series is non-stationary")
    else:
        print("KPSS:       Series is trend stationary")

    if _adf and _kpss:
        print("Conclusion: Series may be difference stationary. Try using differencing.")
    elif (not _adf) and _kpss:
        print("Conclusion: Series is non-stationary")
    elif _adf and (not _kpss):
        print("Conclusion: Series is stationary")
    else:
        print("Conclusion: Series is trend stationary. Try detrending the series.")

    if len(time_series) < 20:
        print("Warning:    Result is not accurate as the series has less than 20 data")
    elif len(time_series) < 45:
        print("Warning:    Having at least 45 data is recommended to use this tester")
    return df


def _adf_test(time_series):
    adf, p_value, used_lag, n_obs, crit_values, best_ic = adfuller(time_series)
    df = pd.Series([adf, p_value], index=['Test Statistic', 'p-value'], name="ADF")
    for k, v in crit_values.items():
        df[f'Critical Value ({k})'] = v
    df["reject_null"] = (p_value <= 0.05)
    return df


def _kpss_test(time_series):
    kpss_, p_value, used_lag, crit_values = kpss(time_series, nlags="auto")
    df = pd.Series([kpss_, p_value], index=['Test Statistic', 'p-value'], name="KPSS")
    for k, v in crit_values.items():
        if k != "2.5%":
            df[f'Critical Value ({k})'] = v
    df["reject_null"] = (p_value <= 0.05)
    return df


def infer_datetime_index(series, last_date, freq):
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    series.index = pd.date_range(last_date, periods=len(series)+1, freq=freq)[1:]
    return series