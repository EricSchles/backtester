import pandas as pd
import datetime
from backtester.metrics import bt_metrics
import code

def describe(
        y_true: pd.Series,
        y_pred: pd.Series,
        start_date: datetime.datetime,
        end_date: datetime.datetime):
    """
    Describes the time series differences against 
    the full set of metrics:
    * unscaled mean bounded relative absolute error
    * geometric mean relative absolute error
    * mean absolute scaled error
    * symmetric mean absolute percentage error
    * median relative absolute error
    * mean relative absolute error
    * root mean squared error
    * mean absolute percentage error
    * mean bounded relative absolute error
    
    Parameters
    ----------
    y_true : pd.Series
        observed values
    y_pred : pd.Series
        forecasted values
    start_date : datetime.datetime
        the start date of the forecast to consider
    end_date : datetime.datetime
        the last date of the forecast to consider

    """
    true_series = true_series[start_date: end_date]
    predicted_series = predicted_series[start_date: end_date]
    if len(true_series) != len(predicted_series):
        raise Exception("""
        For the specified series length of observations does not
        equal length of forecast
        """)
    print(
        "unscaled mean bounded relative absolute error:",
        bt_metrics.umbrae(true_series, predicted_series)
    )
    print(
        "geometric_mean_relative_absolute_error:",
        bt_metrics.gmrae(true_series, predicted_series)
    )
    print(
        "mean_absolute_scaled_error",
        bt_metrics.mase(true_series, predicted_series)
    )
    print(
        "symmetric mean absolute percentage error:",
        bt_metrics.smape(true_series, predicted_series)
    )
    print(
        "median relative absolute error:",
        bt_metrics.median_rae(true_series, predicted_series)
    )
    print(
        "mean relative absolute error:",
        bt_metrics.mean_rae(true_series, predicted_series)
    )
    print(
        "root mean squared error:",
        bt_metrics.rmse(true_series, predicted_series)
    )
    print(
        "mean absolute percentage error",
        bt_metrics.mape(true_series, predicted_series)
    )
    print(
        "mean bounded relative absolute error",  
        bt_metrics.mbrae(true_series, predicted_series)
    )

def analyze_series(timeseries):
    """
    Analyzes the series for specific properties:
    * Ad fuller - presence of unit root.
    If no unit root, series regresses to the mean.
    * KPSS - stationarity
    If stationary regresses to the mean.
    * BDS - time series is independent and iid.
    If so, no serial correlation.
    * Ljung-Box - explicit check for serial correlation.
    If no serial correlation, regresses to the mean.
    
    Parameters
    ----------
    timeseries : pd.Series
        observed values
    """
    print(
        "Ad fuller test",
        "Null Hypothesis: there is a unit root",
        "Alt. Hypothesis: there is no unit root",
        bt_metrics.ad_fuller_test(timeseries)
    )
    print(
        "KPSS test",
        "Null Hypothesis: level stationary",
        "Alt. Hypothesis: not level stationary",
        bt_metrics.kpss(timeseries)
    )
    print(
        "KPSS test",
        "Null Hypothesis: trend stationary",
        "Alt. Hypothesis: not trend stationary",
        bt_merics.kpss(
            timeseries, regression="ct"
        )
    )
    print(
        "BDS test - conservative",
        "epsilon = 1.0",
        "Null Hypothesis: time series is
        independent and iid",
        "Alt. Hypothesis: time series has
        dependence structure over time",
        bt_metrics.bds(
            timeseries, epsilon=1.0
        )
    )
    print(
        "BDS test - liberal",
        "epsilon = 2.0",
        "Null Hypothesis: time series is
        independent and iid",
        "Alt. Hypothesis: time series has
        dependence structure over time",
        bt_metrics.bds(
            timeseries, epsilon=2.0
        )
    )
    print(
        "Ljung-Box Q statistic",
        "Null Hypothesis:",
        "the data is independently distributed",
        "Alt. Hypothesis",
        "the data exhibits serial correlation",
        bt_metrics.q_stat(timeseries)
    )

def analyze_residuals(observations: pd.Series, residuals: pd.Series):
    """
    Analyzes the stability of a forecast against observations.
    Tests progress in degree of instability:
    * Breusch-Godfrey - serial correlation test
    * White - heteroscedasticity test
    * Engle - conditional heteroscedasticity
    * Cumulative Sum - structural break test
    
    Parameters
    ----------
    observations : pd.Series
        observed values
    residuals : pd.Series
        observed values - forecasted values 
    """
    print(
        "Breusch-Godfrey test",
        "number of lags = 1",
        "Null Hypothesis:",
        "There is no serial correlation up to nlags",
        "Alt. Hypothesis:",
        "There is serial correlation",
        bt_metrics.acorr_breusch_godfrey(
            residuals, nlags=1
        )
    )
    print(
        "Breusch-Godfrey test",
        "number of lags = 3",
        "Null Hypothesis:",
        "There is no serial correlation up to nlags",
        "Alt. Hypothesis:",
        "There is serial correlation",
        bt_metrics.acorr_breusch_godfrey(
            residuals, nlags=3
        )
    )
    print(
        "Breusch-Godfrey test",
        "number of lags = 5",
        "Null Hypothesis:",
        "There is no serial correlation up to nlags",
        "Alt. Hypothesis:",
        "There is serial correlation",
        bt_metrics.acorr_breusch_godfrey(
            residuals, nlags=5
        )
    )
    print(
        "White's Heteroscedasticity test",
        "Null Hypothesis:",
        "No heteroscedasticity in the residuals",
        "Alt. Hypothesis:",
        "Heteroscedasticity exists",
        bt_metrics.het_white(
            observations, residuals
        )
    )
    print(
        "Engle's test for",
        "Autoregression Conditional Heteroscedasticity",
        "number of lags = 1",
        "Null Hypothesis: ",
        "No conditional heteroscedasticity",
        "Alt. Hypothesis: ",
        "Conditional heteroscedasticity exists",
        bt_metrics.het_arch(
            residuals, nlags=1
        )
    )
    print(
        "Engle's test for",
        "Autoregression Conditional Heteroscedasticity",
        "number of lags = 3",
        "Null Hypothesis: ",
        "No conditional heteroscedasticity",
        "Alt. Hypothesis: ",
        "Conditional heteroscedasticity exists",
        bt_metrics.het_arch(
            residuals, nlags=3
        )
    )
    print(
        "Engle's test for",
        "Autoregression Conditional Heteroscedasticity",
        "number of lags = 5",
        "Null Hypothesis: ",
        "No conditional heteroscedasticity",
        "Alt. Hypothesis: ",
        "Conditional heteroscedasticity exists",
        bt_metrics.het_arch(
            residuals, nlags=5
        )
    )
    print(
        "Breaks Cumulative summation test",
        "for parameter stability",
        "Null Hypothesis: ",
        "No structural break",
        "Alt. Hypothesis: ",
        "A structural break has occurred",
        bt_metrics.breaks_cumsum(
            residuals
        )
    )

