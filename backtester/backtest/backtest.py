import pandas as pd
import datetime
from backtester.metrics import metrics
import code

def describe(
        true_series: pd.Series,
        predicted_series: pd.Series,
        start_date: datetime.datetime,
        end_date: datetime.datetime):
    """
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
        umbrae(true_series, predicted_series)
    )
    print(
        "geometric_mean_relative_absolute_error:",
        gmrae(true_series, predicted_series)
    )
    print(
        "mean_absolute_scaled_error",
        mase(true_series, predicted_series)
    )
    print(
        "symmetric mean absolute percentage error:",
        smape(true_series, predicted_series)
    )
    print(
        "median relative absolute error:",
        median_rae(true_series, predicted_series)
    )
    print(
        "mean relative absolute error:",
        mean_rae(true_series, predicted_series)
    )
    print(
        "root mean squared error:",
        rmse(true_series, predicted_series)
    )
    print(
        "mean absolute percentage error",
        mape(true_series, predicted_series)
    )
    print(
        "mean bounded relative absolute error",  
        mbrae(true_series, predicted_series)
    )

