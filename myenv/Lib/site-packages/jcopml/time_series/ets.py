import pickle
from pathlib import Path
from textwrap import dedent
from itertools import product
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from jcopml.time_series.metric import SCORER
from jcopml.time_series.utils import infer_datetime_index
from jcopml.time_series.warnings import ignore_warning


class AutoETS:
    """
    ETS Forecasting with Grid Search and Cross Validation

    
    == Example usage ==
    model = AutoETS(y_train, y_test, seasonal_periods=12, scoring="rmse").fit()


    == Arguments ==
    series: pandas Series
        time series data

    test_size: int or float
        if int, used as how many data reserved for test set
        if float, used as percentage of data reserved for test set

    seasonal_periods: {int, None}
        Periodicity of the sequence. For seasonal, it should be 12.

    cv: int
        Number of cross validation fold

    scoring: str
        If None, mean squared error would be used. Here are list of supported scorer:
        - mse   Mean Squared Error
        - mae   Mean Absolute Error
        - mape  Mean Absolute Percentage Error
        - msle  Mean Squared Log Error
        - rmse  Root Mean Squared Error

    freq: str, Timedelta, datetime.timedelta, or DateOffset
        If auto, will infer with pd.infer_freq
        else, You could use pandas frequency string or other pandas compatible frequency
        https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases        

    trend: {str, list, None}
        if "auto", then it will be tuned using Grid Search. Here are list of supported trend:
        - add   Additive Trend
        - mul   Multiplicative Trend
        - None  Not using any Trend

        You can also use list of the trend you would want to search in parameter tuning.

    seasonal: {str, list, None}
        if "auto", then it will be tuned using Grid Search. Here are list of supported seasonal:
        - add   Additive Trend
        - mul   Multiplicative Trend
        - None  Not using any Trend

        You can also use list of the seasonal you would want to search in parameter tuning.

    damped: {str, bool}
        if "auto", then it will be tuned using Grid Search.
        if True, then the ETS would utilize damping.

    boxcox: {str, bool}
        if "auto", then it will be tuned using Grid Search.
        if True, then the series would be transformed with boxcox.
    """    
    def __init__(self, series, test_size, cv=4, scoring="auto", freq="auto", seasonal_periods=None, trend="auto", seasonal="auto", damped="auto", boxcox="auto"):
        self.y_train, self.y_test = self.split_data(series, test_size)
        self.last_date = self.get_last_date()
        self.freq = pd.infer_freq(series.index) if freq == "auto" else freq
        
        self.cv = cv
        self.scoring, self.scorer = self._process_scoring(scoring)

        self.trend = self._process_trend(trend)
        self.seasonal_periods = self._process_seasonal_periods(seasonal_periods)
        self.seasonal = self._process_seasonal(seasonal_periods, seasonal)
        self.damped = self._process_damped(damped)
        self.boxcox = self._process_boxcox(boxcox)
        
    @ignore_warning
    def fit(self, ignore_error=True):
        series = self.y_train
        cv_results = defaultdict(lambda: [])
        
        param_grid = list(product(self.trend, self.seasonal_periods, self.seasonal, self.damped, self.boxcox))
        for trend, s_period, seasonal, damped, boxcox in tqdm(param_grid, leave=False):
            if (trend is None) and damped:
                # There is no such thing as damped on no trend
                continue
            
            try:
                for fold, (train_id, val_id) in enumerate(TimeSeriesSplit(n_splits=self.cv).split(series)):
                    # CV split
                    train, val = series.iloc[train_id], series.iloc[val_id]

                    # Train
                    es = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, damped_trend=damped,
                                              seasonal_periods=s_period, use_boxcox=boxcox, initialization_method=None)
                    model = es.fit(optimized=True)

                    # Validate
                    start, end = val.index[0], val.index[-1]
                    pred = model.predict(start, end)
                    score = self.scorer(val, pred)
                    cv_results[f"{self.scoring}_{fold}"].append(score)

                # Reporting
                cv_results["trend"].append(trend)
                cv_results["seasonal"].append(seasonal)
                cv_results["seasonal_periods"].append(s_period)
                cv_results["damped_trend"].append(damped)
                cv_results["use_boxcox"].append(boxcox)
            except Exception as e:
                if not ignore_error:
                    print(e)
                continue
        param_cols = ["trend", "seasonal", "seasonal_periods", "damped_trend", "use_boxcox"]
        result_cols = [f"{self.scoring}_{i}" for i in range(self.cv)]
        
        cv_results = pd.DataFrame(cv_results)
        cv_results = cv_results[param_cols + result_cols]
        cv_results[f"mean_{self.scoring}"] = cv_results.loc[:, result_cols].mean(1)
        cv_results.sort_values(f"mean_{self.scoring}", inplace=True)
        cv_results.reset_index(drop=True, inplace=True)

        self.cv_results = cv_results
        self.best_params = cv_results.loc[0, param_cols].to_dict()
        self.refit()
        return self
        
    def plot(self, n_prior=0, figsize=(15, 3)):
        model = self.model_
        y_train = self.y_train
        y_test = self.y_test
        score = {}
        
        train_result = y_train.to_frame(name="series").copy()
        train_result["fitted"] = model.fittedvalues
        score["train"] = self.scorer(train_result.series, train_result.fitted)
        
        n_forecast = len(y_test) + n_prior
        if n_forecast == 0:
            test_result = pd.DataFrame({"series": [], "fitted": []})
        else:
            test_result = model.forecast(n_forecast).to_frame(name="fitted").copy()
            if not y_test.empty:
                test_result["series"] = y_test                
                score["test"] = self.scorer(test_result.dropna().series, test_result.dropna().fitted)
        
        if test_result.empty:
            result = train_result
        else:
            result = pd.concat([train_result, test_result])

        plt.figure(figsize=figsize)
        plt.plot(train_result.series, "b-", label="train")
        plt.plot(train_result.fitted, "r--", label="fitted")
        if "series" in test_result:
            plt.plot(test_result.series, "k-", label="test")
        if "fitted" in test_result:
            plt.plot(test_result.fitted, "mo--", markersize=3, label="forecast")
        plt.title(" | ".join([f"{k.title()} {self.scoring.upper()} = {v:.3f}" for k, v in score.items()]))
        ylim = self._get_ylim(result)
        if ylim:
            plt.ylim(*ylim)
        plt.legend(loc="upper left");
        return result, score
        
    def forward(self, n_data=None):
        if (n_data is None) or (n_data == len(self.y_test)):
            self.update(self.y_test)
        else:
            n_data = min(len(self.y_test), n_data)
            new_data = self.y_test.iloc[:n_data]
            prior_data = self.y_test.iloc[n_data:]
            self.update(new_data, prior_data)
    
    def update(self, series, test_series=None):
        series = infer_datetime_index(series, self.last_date, self.freq)
        if test_series is not None:
            test_series = infer_datetime_index(test_series, series.index[-1], self.freq)
        
        self.y_train = pd.concat([self.y_train, series])
        self.last_date = self.get_last_date()
        self.y_test = self.y_train.iloc[0:0] if test_series is None else test_series
        
        self.refit()
        print(f"Data is updated to {self.last_date}")
    
    def save(self, filepath):
        fpath = Path(filepath)
        fpath.parent.mkdir(exist_ok=True, parents=True)
        pickle.dump(self, open(fpath, "wb"))
        print(f"Model is saved to {fpath}")

    @ignore_warning
    def refit(self):
        es = ExponentialSmoothing(self.y_train, initialization_method=None, **self.best_params)
        self.model_ = es.fit(optimized=True)
        return self.model_
                
    def get_last_date(self):
         return self.y_train.index[-1]
    
    def split_data(self, series, test_size):
        if isinstance(test_size, int) and (test_size < len(series)):
            return series.iloc[:-test_size], series.iloc[-test_size:]
        elif isinstance(test_size, float) and (0 < test_size < 1):
            return self.split_data(series, int(test_size * len(series)))
        
    @staticmethod
    def _process_trend(trend):
        if trend == "auto":
            return [None, "add", "mul"]
        elif isinstance(trend, list):
            return trend
        elif isinstance(trend, str) or (trend is None):
            return [trend]

    @staticmethod        
    def _process_seasonal_periods(seasonal_periods):
        if (seasonal_periods is None) or isinstance(seasonal_periods, int):
            return [seasonal_periods]
        elif isinstance(seasonal_periods, list):
            return seasonal_periods

    @staticmethod        
    def _process_seasonal(seasonal_periods, seasonal):
        if seasonal_periods is None:
            return [None]
        else:
            if (seasonal == "auto") or (seasonal is None):
                return ["add", "mul"]
            elif isinstance(seasonal, list):
                return [s for s in seasonal if s in ["add", "mul"]]
            elif isinstance(seasonal, str):
                assert seasonal in ["add", "mul"]
                return [seasonal]
        
    @staticmethod        
    def _process_damped(damped):        
        if damped == "auto":
            return [True, False]
        elif isinstance(damped, bool):
            return [damped]

    @staticmethod
    def _process_boxcox(boxcox):        
        if boxcox == "auto":
            return [True, False]
        elif isinstance(boxcox, bool):
            return [boxcox]       

    @staticmethod        
    def _process_scoring(scoring):
        if scoring not in SCORER:
            scoring = "rmse"
        return scoring, SCORER[scoring]
    
    @staticmethod
    def _get_ylim(result):
        min_data, max_data = result.series.min(), result.series.max()
        data_range = max_data - min_data

        ymin, ymax = min_data - 0.05 * data_range, max_data + 0.05 * data_range

        if (result.fitted.min() < ymin) or (result.fitted.max() > ymax):
            return (ymin, ymax)


def auto_ets(series, seasonal_periods=None, cv=5, scoring=None, trend="auto", seasonal="auto", damped="auto",
             boxcox="auto"):
    """
    Deprecated in favor of AutoETS since jcopml 1.2.4
    """
    deprecation_message = """
    auto_ets is deprecated since jcopml 1.2.4. Please use AutoETS instead.
    >> from jcopml.time_series import AutoETS
    """
    raise Exception(dedent(deprecation_message)[1:-1])
