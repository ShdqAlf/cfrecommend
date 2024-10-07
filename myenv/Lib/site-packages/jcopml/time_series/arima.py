import pickle
from pathlib import Path
from itertools import product
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from pmdarima.arima import ndiffs, nsdiffs
from pmdarima import AutoARIMA as PyramidAutoARIMA, ARIMA as PyramidARIMA
from pmdarima.pipeline import Pipeline, check_endog, DTYPE, clone
from pmdarima.preprocessing import BoxCoxEndogTransformer, LogEndogTransformer
from sklearn.model_selection import TimeSeriesSplit

from jcopml.time_series.metric import SCORER
from jcopml.time_series.utils import infer_datetime_index
from jcopml.time_series.warnings import ignore_warning


class ArimaPipeline(Pipeline):
    def __init__(self, steps):
        super().__init__(steps)
        
    def fit(self, y, X=None, **fit_kwargs):
        """Fit the pipeline of transformers and the ARIMA model

        Chain the time-series and X array through a series of
        transformations, fitting each stage along the way, finally fitting an
        ARIMA or AutoARIMA model.

        Parameters
        ----------
        y : array-like or iterable, shape=(n_samples,)
            The time-series to which to fit the ``ARIMA`` estimator. This may
            either be a Pandas ``Series`` object (statsmodels can internally
            use the dates in the index), or a numpy array. This should be a
            one-dimensional array of floats, and should not contain any
            ``np.nan`` or ``np.inf`` values.

        X : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.

        **fit_kwargs : keyword args
            Extra keyword arguments used for each stage's ``fit`` stage.
            Similar to scikit-learn pipeline keyword args, the keys are
            compound, comprised of the stage name and the argument name
            separated by a "__". For instance, if fitting an ARIMA in stage
            "arima", your kwargs may resemble::

                {"arima__maxiter": 10}
        """
        # Shallow copy
        steps = self.steps_ = self._validate_steps()

        yt = check_endog(y, dtype=DTYPE, copy=False, preserve_series=True)
        Xt = X
        named_kwargs = self._get_kwargs(**fit_kwargs)

        # store original shape for later in-sample preds
        self.n_samples_ = yt.shape[0]

        for step_idx, name, transformer in self._iter(with_final=False):
            cloned_transformer = clone(transformer)
            kwargs = named_kwargs[name]
            yt, Xt = cloned_transformer.fit_transform(yt, Xt, **kwargs)

            # Replace the transformer of the step with the fitted
            # transformer.
            steps[step_idx] = (name, cloned_transformer)

        # Save the order of the columns so we can select in the predict phase
        self.x_feats_ = Xt.columns.tolist() \
            if isinstance(Xt, pd.DataFrame) else None

        # Now fit the final estimator
        kwargs = named_kwargs[steps[-1][0]]
        estimator = self._final_estimator
        estimator.fit(yt, X=Xt, **kwargs)
        
        if isinstance(estimator, PyramidAutoARIMA):
            self.order = estimator.model_.order
            self.seasonal_order = estimator.model_.seasonal_order
        elif isinstance(estimator, PyramidARIMA):
            self.order = estimator.order
            self.seasonal_order = estimator.seasonal_order
        self.name = f"SARIMAX{self.order}{self.seasonal_order}"
        return self        
    

class AutoARIMA:
    """
    ARIMA Forecasting with Grid Search and Cross Validation


    == Example usage ==
    model, cv_results = auto_ets(series, seasonal_periods=12)


    == Arguments ==
    series: pandas Series
        time series data

    test_size: int or float
        if int, used as how many data reserved for test set
        if float, used as percentage of data reserved for test set

    exog: pandas Dataframe
        exogenous time series feature

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

    seasonal_periods: {int, None}
        Periodicity of the sequence. For seasonal, it should be 12.

    d: "auto", List[int], int
        non-seasonal differencing. If "auto", then it will try to estimate d with kpss and adf test

    D: "auto", List[int], int
        Seasonal differencing. If "auto", then it will try to estimate D with ocsb and ch test

    transform: "auto", List[str|None], str, None
        Data transformation. "boxcox" and "log" is supported. If "auto", then it will be tuned with Grid Search.

    information_criterion: {aic, aicc, bic, hqic}
        information criterion used for step-wise optimization in pmdarima

    random_state: int
        random state value
    """        
    def __init__(self, series, test_size, exog=None, cv=4, scoring="auto", freq="auto", seasonal_periods=None, d="auto", D="auto", transform="auto", information_criterion="aic", random_state=42):
        self.y_train, self.y_test = self.split_data(series, test_size)
        self.has_exog = exog is not None
        if self.has_exog:
            self.X_train, self.X_test = self.split_data(exog, test_size)
        else:
            self.X_train = self.X_test = None
        self.last_date = self.get_last_date()
        self.freq = pd.infer_freq(series.index) if freq == "auto" else freq
        
        self.cv = cv
        self.scoring, self.scorer = self._process_scoring(scoring)
        
        self.m = self._process_seasonal_periods(seasonal_periods)
        self.seasonal = self._process_seasonal(seasonal_periods)        
        self.d = self._process_d(d)
        self.D = self._process_D(D)   
        self.transform = self._process_transform(transform)
        self.pipeline_config = {
            "information_criterion": information_criterion,
            "random_state": random_state
        }        
        self.information_criterion = information_criterion
        self.random_state = random_state

    @ignore_warning
    def fit(self, ignore_error=True):        
        series = self.y_train
        X = self.X_train
        cv_results = defaultdict(lambda: [])
        
        param_grid = list(product(self.d, self.D, self.m))
        for d, D, m in tqdm(param_grid, leave=False):
            try:
                for fold, (train_id, val_id) in enumerate(TimeSeriesSplit(n_splits=self.cv).split(series)):
                    # CV split
                    train, val = series.iloc[train_id], series.iloc[val_id]
                    if self.has_exog:
                        X_train, X_val = X.iloc[train_id], X.iloc[val_id]
                    else:
                        X_train = X_val = None
                        
                    # Train
                    arima = create_auto_arima_pipeline(None, d, D, self.seasonal, m, **self.pipeline_config)
                    model = arima.fit(train, X=X_train)

                    # Validate
                    pred = model.predict(len(val), X=X_val)
                    score = self.scorer(val, pred)
                    cv_results[f"{self.scoring}_{fold}"].append(score)

                # Reporting
                cv_results["d"].append(d)
                cv_results["D"].append(D)
                cv_results["transform"].append(None)
                cv_results["seasonal"].append(self.seasonal)
                cv_results["m"].append(m)
                cv_results["order"].append(model.order)
                cv_results["seasonal_order"].append(model.seasonal_order)
            except Exception as e:
                if not ignore_error:
                    print(e)
                continue
                
        param_cols = ["d", "D", "transform", "seasonal", "m", "order", "seasonal_order"]
        result_cols = [f"{self.scoring}_{i}" for i in range(self.cv)]                
        
        results = pd.DataFrame(cv_results).copy()
        results = results[param_cols + result_cols]
        results[f"mean_{self.scoring}"] = results.loc[:, result_cols].mean(1)
        results.sort_values(f"mean_{self.scoring}", inplace=True)
        results.reset_index(drop=True, inplace=True)
        best_params = results.loc[0, param_cols].to_dict()
        for transform in tqdm(self.transform, leave=False):
            try:
                for fold, (train_id, val_id) in enumerate(TimeSeriesSplit(n_splits=self.cv).split(series)):
                    # CV split
                    train, val = series[train_id], series[val_id]
                    if self.has_exog:
                        X_train, X_val = X.iloc[train_id], X.iloc[val_id]
                    else:
                        X_train = X_val = None
                        
                    # Train
                    best_params["transform"] = transform
                    arima = create_auto_arima_pipeline(**best_params, **self.pipeline_config)
                    model = arima.fit(train, X=X_train)

                    # Validate
                    pred = model.predict(len(val), X=X_val)
                    score = self.scorer(val, pred)
                    cv_results[f"{self.scoring}_{fold}"].append(score)

                # Reporting
                cv_results["d"].append(best_params["d"])
                cv_results["D"].append(best_params["D"])
                cv_results["transform"].append(transform)
                cv_results["seasonal"].append(best_params["seasonal"])
                cv_results["m"].append(best_params["m"])
                cv_results["order"].append(model.order)
                cv_results["seasonal_order"].append(model.seasonal_order)                
            except Exception as e:
                if not ignore_error:
                    print(e)
                continue
                
        cv_results = pd.DataFrame(cv_results)
        cv_results = cv_results[param_cols + result_cols]
        cv_results[f"mean_{self.scoring}"] = cv_results.loc[:, result_cols].mean(1)
        cv_results.sort_values(f"mean_{self.scoring}", inplace=True)
        cv_results.reset_index(drop=True, inplace=True)

        self.cv_results = cv_results
        self.best_params = cv_results.loc[0, param_cols].to_dict()

        arima = create_auto_arima_pipeline(**self.best_params, **self.pipeline_config)
        self.model_ = arima.fit(self.y_train, X=self.X_train)
        self.order = self.model_.order
        self.seasonal_order = self.model_.seasonal_order
        self.best_params["order"] = self.order
        self.best_params["seasonal_order"] = self.seasonal_order

        self.refit()
        print(f"Found optimal model: {self.model_.name}")
        return self

    def plot(self, n_prior=0, exog_prior=None, start=0, figsize=(15, 3)):
        model = self.model_
        y_train, y_test = self.y_train, self.y_test
        if self.has_exog:
            X_train, X_test = self.X_train, self.X_test
            if exog_prior is not None:
                X_prior = pd.DataFrame(exog_prior, columns=model.x_feats_)
                last_date = self.last_date if len(y_test) == 0 else y_test.index[-1]
                X_prior = infer_datetime_index(X_prior, last_date, self.freq)
                X_test = pd.concat([X_test, X_prior])            
        else:
            X_train = X_test = None
        score = {}
        
        train_result = y_train.to_frame(name="series").copy()
        train_result["fitted"] = model.predict_in_sample(X=X_train)
        score["train"] = self.scorer(train_result.series, train_result.fitted)
        
        n_forecast = len(y_test) + n_prior
        if self.has_exog:
            n_forecast = min(n_forecast, len(X_test))

        if n_forecast == 0:
            test_result = pd.DataFrame({"series": [], "fitted": []})
        else:
            test_result = model.predict(n_forecast, X=X_test)
            test_result = infer_datetime_index(test_result, self.last_date, self.freq)
            test_result = test_result.to_frame("fitted").copy()
            if not y_test.empty:
                test_result["series"] = y_test
                score["test"] = self.scorer(test_result.dropna().series, test_result.dropna().fitted)
        
        if self.has_exog:
            train_result = pd.concat([train_result, X_train], axis=1)
            test_result = pd.concat([test_result, X_test], axis=1)

        if test_result.empty:
            result = train_result
        else:
            result = pd.concat([train_result, test_result])    

        plt.figure(figsize=figsize)
        plt.plot(train_result.series.iloc[start:], "b-", label="train")
        plt.plot(train_result.fitted.iloc[start:], "r--", label="fitted")
        if "series" in test_result:
            plt.plot(test_result.series.iloc[start:], "k-", label="test")
        if "fitted" in test_result:
            plt.plot(test_result.fitted.iloc[start:], "mo--", markersize=3, label="forecast")
        plt.title(" | ".join([f"{k.title()} {self.scoring.upper()} = {v:.3f}" for k, v in score.items()]))
        ylim = self._get_ylim(result.iloc[start:])
        if ylim:
            plt.ylim(*ylim)
        plt.legend(loc="upper left");
        return result, score

    def forward(self, n_data=None):
        if (n_data is None) or (n_data == len(self.y_test)):
            self.update(self.y_test, self.X_test)
        else:
            n_data = min(len(self.y_test), n_data)
            new_data = self.y_test.iloc[:n_data]
            prior_data = self.y_test.iloc[n_data:]
            if self.has_exog:
                new_exog_data = self.X_test.iloc[:n_data]
                prior_exog_data = self.X_test.iloc[n_data:]
            else:
                new_exog_data = prior_exog_data = None
            self.update(new_data, new_exog_data, prior_data, prior_exog_data)
    
    def update(self, series, exog=None, test_series=None, test_exog=None):
        series = infer_datetime_index(series, self.last_date, self.freq)
        exog = pd.DataFrame(exog, columns=self.model_.x_feats_)        
        exog = infer_datetime_index(exog, self.last_date, self.freq)
        if test_series is not None:
            test_series = infer_datetime_index(test_series, series.index[-1], self.freq)
        if test_exog is not None:
            test_exog = infer_datetime_index(test_exog, exog.index[-1], self.freq)

        self.y_train = pd.concat([self.y_train, series])
        self.y_test = self.y_train.iloc[0:0] if test_series is None else test_series
        if self.has_exog:
            self.X_train = pd.concat([self.X_train, exog])
            self.X_test = self.X_train.iloc[0:0] if test_exog is None else test_exog
        self.last_date = self.get_last_date()            
        
        self.refit()
        print(f"Data is updated to {self.last_date}")    
    
    def save(self, filepath):
        fpath = Path(filepath)
        fpath.parent.mkdir(exist_ok=True, parents=True)
        pickle.dump(self, open(fpath, "wb"))
        print(f"Model is saved to {fpath}")

    @ignore_warning
    def refit(self):
        arima = create_arima_pipeline(**self.best_params)
        self.model_ = arima.fit(self.y_train, X=self.X_train)
        return self.model_

    def get_last_date(self):
         return self.y_train.index[-1]    
    
    def split_data(self, series, test_size):
        if isinstance(test_size, int) and (test_size < len(series)):
            return series.iloc[:-test_size], series.iloc[-test_size:]
        elif isinstance(test_size, float) and (0 < test_size < 1):
            return self.split_data(series, int(test_size * len(series)))       
    
    @staticmethod        
    def _process_seasonal_periods(seasonal_periods):
        if seasonal_periods is None:
            return [0]
        elif isinstance(seasonal_periods, int):
            return [seasonal_periods]
        elif isinstance(seasonal_periods, list):
            return seasonal_periods
        
    @staticmethod
    def _process_seasonal(seasonal_periods):
        return seasonal_periods is not None
        
    def _process_d(self, d):
        if d == "auto":
            kpss_diffs = ndiffs(self.y_train, alpha=0.05, test='kpss', max_d=6)
            adf_diffs = ndiffs(self.y_train, alpha=0.05, test='adf', max_d=6)
            d = max(adf_diffs, kpss_diffs)
            return range(d+1)
        elif isinstance(d, int):
            return [d]
        elif isinstance(d, list):
            return d
        
    def _process_D(self, D):
        if not self.seasonal:
            return [0]
        
        if D == "auto":
            ocsb_diffs = [nsdiffs(self.y_train, m=m, test='ocsb', max_D=6) for m in self.m]
            ch_diffs = [nsdiffs(self.y_train, m=m, test='ch', max_D=6) for m in self.m]
            D = max(*ocsb_diffs, *ch_diffs)
            return range(D+1)
        elif isinstance(D, int):
            return [D]
        elif isinstance(D, list):
            return D
        
    @staticmethod    
    def _process_transform(transform):        
        if transform == "auto":
            return ["boxcox", "log"]
        elif isinstance(transform, str) or (transform is None):
            return [transform]
        elif isinstance(transform, list):
            return transform
        
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


def create_auto_arima_pipeline(transform, d, D, seasonal, m, information_criterion, random_state=42, **kwargs):
    if transform == "boxcox":
        pipe = ArimaPipeline([
            ('prep', BoxCoxEndogTransformer(lmbda2=1e-6, neg_action='ignore')),
            ('arima', PyramidAutoARIMA(d=d, D=D, information_criterion=information_criterion, seasonal=seasonal, m=m, random_state=random_state))
        ])
    elif transform == "log":
        pipe = ArimaPipeline([
            ('prep', LogEndogTransformer(lmbda=1e-6, neg_action='ignore')),
            ('arima', PyramidAutoARIMA(d=d, D=D, information_criterion=information_criterion, seasonal=seasonal, m=m, random_state=random_state))
        ])
    else:
        pipe = ArimaPipeline([
            ('arima', PyramidAutoARIMA(d=d, D=D, information_criterion=information_criterion, seasonal=seasonal, m=m, random_state=random_state))
        ]) 
    return pipe


def create_arima_pipeline(transform, order, seasonal_order, **kwargs):
    if transform == "boxcox":
        pipe = ArimaPipeline([
            ('prep', BoxCoxEndogTransformer(lmbda2=1e-6, neg_action='ignore')),
            ('arima', PyramidARIMA(order, seasonal_order))
        ])
    elif transform == "log":
        pipe = ArimaPipeline([
            ('prep', LogEndogTransformer(lmbda=1e-6, neg_action='ignore')),
            ('arima', PyramidARIMA(order, seasonal_order))
        ])
    else:
        pipe = ArimaPipeline([
            ('arima', PyramidARIMA(order, seasonal_order))
        ]) 
    return pipe
