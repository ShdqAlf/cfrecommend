import pandas as pd
import matplotlib.pyplot as plt
from .metric import _get_scorer
from matplotlib.collections import PolyCollection

from pmdarima.base import BaseARIMA
from pmdarima.pipeline import Pipeline as ArimaPipeline
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_fitted(model, y_train, scoring="rmse", figsize=(15, 3)):
    """
    Visualize fitted statsmodels.tsa model


    == Example usage ==
    plot_fitted(model, y_train, scoring="rmse")


    == Arguments ==
    y_train: pandas Series
        time series data

    model: statsmodels.tsa model
        fitted statsmodels.tsa model

    scoring: str
        If None, mean squared error would be used. Here are list of supported scorer:
        - mse   Mean Squared Error
        - mae   Mean Absolute Error
        - mape  Mean Absolute Percentage Error
        - msle  Mean Squared Log Error
        - rmse  Root Mean Squared Error
    """
    result = y_train.to_frame(name="series").copy()
    result["fitted"] = model.fittedvalues
    scoring, scorer = _get_scorer(scoring)
    score = scorer(result.series, result.fitted)

    plt.figure(figsize=figsize)
    plt.plot(result.series, "b-", label="train")
    plt.plot(result.fitted, "ro--", markersize=3, label="fitted")
    plt.title(f"{scoring.upper()} = {score:.3f}")
    plt.legend(loc="upper left");
    return result, score
    

def plot_forecast(model, y_train, y_test, X_train=None, X_test=None, n_prior=0, scoring="rmse", figsize=(15, 3)):
    """
    Example Usage
    result, score = plot_forecast(model, y_train, y_test, n_prior=12)
    """
    train_result = y_train.to_frame(name="series").copy()
    if isinstance(model, (BaseARIMA, ArimaPipeline)):
        train_result["fitted"] = model.predict_in_sample(X=X_train)
    else:
        train_result["fitted"] = model.fittedvalues
    
    n_forecast = len(y_test) + n_prior
    if isinstance(model, (BaseARIMA, ArimaPipeline)):
        test_result = model.predict(n_forecast, X=X_test).to_frame(name="fitted").copy()
    else:
        test_result = model.forecast(n_forecast).to_frame(name="fitted").copy()
    test_result["series"] = y_test
    
    if isinstance(model, (BaseARIMA, ArimaPipeline)) and (X_train is not None):
        train_result = pd.concat([train_result, X_train], axis=1)
        test_result = pd.concat([test_result, X_test], axis=1)
    
    scoring, scorer = _get_scorer(scoring)
    train_score = scorer(train_result.series, train_result.fitted)    
    test_score = scorer(test_result.dropna().series, test_result.dropna().fitted)

    result = pd.concat([train_result, test_result])
    score = {"train": train_score, "test": test_score}

    plt.figure(figsize=figsize)
    if isinstance(model, (BaseARIMA, ArimaPipeline)):
        n_skip = max(model.order[1], model.seasonal_order[-1])
        plt.plot(train_result.series.iloc[n_skip:], "b-", label="train")
        plt.plot(train_result.fitted.iloc[n_skip:], "r--", label="fitted")
    else:
        plt.plot(train_result.series, "b-", label="train")
        plt.plot(train_result.fitted, "r--", label="fitted")
    plt.plot(test_result.series, "k-", label="test")
    plt.plot(test_result.fitted, "mo--", markersize=3, label="forecast")
    plt.title(f"Train {scoring.upper()} = {train_score:.3f} | Test {scoring.upper()} = {test_score:.3f}")
    ylim = _get_ylim(result)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(loc="upper left");
    return result, score


def plot_acf_pacf(series, figsize=(15, 8), max_lag=50, dropna=True):
    if dropna:
        series = series.dropna()

    n_sample = len(series)

    plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.plot(series, "b-")

    ax1 = plt.subplot(223)
    plot_pacf(series, lags=min(n_sample, max_lag) // 2 - 1, ax=ax1, title="PACF (for AR)", color="r", vlines_kwargs={"colors": "r"}, alpha=0.05);
    for item in ax1.collections:
        if type(item) == PolyCollection:
            item.set_facecolor('r')
            item.set_alpha(0.15)

    ax2 = plt.subplot(224)
    plot_acf(series, lags=min(n_sample, max_lag) - 1, ax=ax2, title="ACF (for MA)", color="b", vlines_kwargs={"colors": "b"}, alpha=0.05);
    for item in ax2.collections:
        if type(item) == PolyCollection:
            item.set_facecolor('b')
            item.set_alpha(0.15)


def plot_lag_correlation(series_a, series_b, n_lag=12, figsize=(12, 3)):
    """
    Performing correlation on multiple lag
    series A is static
    series B is shifted

    Example Usage:
    df_lag, fig = plot_lag_correlation(df.column_a, df.column_b)
    """
    df = [(i, series_a.corr(series_b.shift(i))) for i in range(-n_lag, n_lag+1)]
    df = pd.DataFrame(df, columns=["lag", "correlation"])
    
    diverge_cmap = plt.get_cmap("bwr_r")
    barcolor = diverge_cmap(df.correlation.apply(lambda x: x + 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    plt.grid(axis="y", alpha=0.5)
    ax.bar(df.lag, df.correlation, color=barcolor, zorder=3)
    ax.set(ylim=(-1, 1), xlabel="Lag", ylabel="corr")

    best = df.loc[df.correlation.abs().argmax()]
    plt.annotate('*', xy=(best.lag, best.correlation), xytext=(0, 0),
                 textcoords='offset points', ha='center', va='center', fontsize=18, color="k")
    plt.title(f"Most correlated at lag {best.lag:.0f} [{best.correlation:.2f}]")
    return df, fig


def _get_ylim(result):
    min_data, max_data = result.series.min(), result.series.max()
    data_range = max_data - min_data

    ymin, ymax = min_data - 0.05 * data_range, max_data + 0.05 * data_range

    if (result.fitted.min() < ymin) or (result.fitted.max() > ymax):
        return (ymin, ymax)
