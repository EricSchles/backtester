import pandas as pd
import datetime
from backtester.metrics import bt_metrics
import code

def describe(
        true_series: pd.Series,
        predicted_series: pd.Series,
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

        
        
