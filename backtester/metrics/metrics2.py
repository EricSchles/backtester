import math
import numpy as np
import pandas as pd
from statsmodels.tsa.api import (
    adfuller, bds, coint, kpss, acf, q_stat
)
from statsmodels.stats import diagnostic
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import namedtuple
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import metrics
from scipy.stats import mstats
from scipy.stats import iqr
from empiricaldist import Cdf as CDF
from scipy import linalg

def _relative_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Relative Error """
    benchmark = seasonal_decompose(y_true).seasonal.max()
    if benchmark is None or benchmark.is_integer():
        if benchmark == 0:
            seasonality = 1
        else:
            seasonality = benchmark
        error = y_true[seasonality:] - y_pred[seasonality:]
        naive_forecast_error = y_true[seasonality:] - y_true[:-seasonality]  
        return error / naive_forecast_error
    return (y_true - y_pred) / y_true

def _bounded_relative_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Bounded Relative Error """
    benchmark = seasonal_decompose(y_true).seasonal.max()
    if benchmark is None or benchmark.is_integer():
        # If no benchmark prediction provided - use naive forecasting
        if benchmark == 0:
            seasonality = 1
        else:
            seasonality = benchmark
        error = y_true[seasonality:] - y_pred[seasonality:]
        abs_err = np.abs(error)
        naive_forecast_error = y_true[seasonality:] - y_true[:-seasonality]
        abs_error_bench = np.abs(naive_forecast_error)
    else:
        error = y_true[seasonality:] - y_pred[seasonality:]
        abs_err = np.abs(error)
        abs_err_bench = np.abs(y_true - benchmark)
    return abs_err / (abs_err + abs_err_bench)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Root Mean Squared Error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Root Mean Squared Error
    """
    return np.sqrt(
        metrics.mean_squared_error(y_true, y_pred)
    )

def normalized_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Normalized Root Mean Squared Error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Normalized Root Mean Squared Error
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    normalizing_factor = np.abs(y_true.max() - y_pred.min())
    return rmse / normalizing_factor

def mean_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean Error
    """
    return np.mean(y_true - y_pred)

def absolute_error(y_true: pd.Series, y_pred: pd.Series) -> np.array:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Mean_absolute_error
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Absolute error
    """
    return np.abs(y_true.values - y_pred.values)

def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Mean_absolute_error
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean absolute error
    """
    return np.mean(
        absolute_error(y_true, y_pred)
    )

def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Mean_absolute_error
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean absolute error
    """
    return np.mean(
        absolute_error(y_true, y_pred)
    )

def median_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Median absolute error
    """
    return np.median(
        absolute_error(y_true, y_pred)
    )

def variance_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Variance in absolute error
    """
    return np.var(
        absolute_error(y_true, y_pred)
    )

def iqr_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Interquartile Range in absolute error
    """
    return iqr(
        absolute_error(y_true, y_pred)
    )

mad = mae  # Mean Absolute Deviation (it is the same as MAE)

def geometric_mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    formula comes from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Geometric Mean Absolute Percentage Error
    """
    return mstats.gmean(
        absolute_error(y_true, y_pred)
    )

def mean_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Percentage Error
    Formula taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean percentage error as a float.
    """

    return np.mean(
        (y_true - y_pred) / y_true
    )

def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float: 
    """
    Mean Absolute Percentage Error
    Formula taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean absolute percentage error as a float.
    """
    percentage_error = (y_true.values - y_pred.values) / y_true.values
    absolute_percentage_error = np.abs(percentage_error)
    return np.mean(absolute_percentage_error) * 100

def median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Median Absolute Percentage Error
    Formula taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean absolute percentage error as a float.
    """

    return np.median(np.abs(
        (y_true - y_pred) / y_true
    ))

def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Symmetric Mean Absolute Percentage Error
    """
    numerator = np.abs(y_pred.values - y_true.values)
    denominator = np.abs(y_true.values) + np.abs(y_pred.values)
    denominator /= 2
    return np.mean(numerator / denominator) * 100

def symmetric_median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Symmetric Mean Absolute Percentage Error
    """
    numerator = np.abs(y_pred.values - y_true.values)
    denominator = np.abs(y_true.values) + np.abs(y_pred.values)
    denominator /= 2
    return np.median(numerator / denominator) * 100

def mean_arctangent_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    relative_error = (actual - predicted) / (actual)
    abs_relative_error = np.abs(relative_error)
    return np.mean(np.arctan(
        abs_relative_error
    ))

def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Absolute Scaled Error

    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    benchmark = seasonal_decompose(y_true).seasonal.max()
    benchmark = math.floor(benchmark)
    # If no benchmark prediction provided - use naive forecasting
    if benchmark == 0:
        seasonality = 1
    else:
        seasonality = benchmark
    error = y_true[seasonality:] - y_pred[seasonality:]
    naive_forecast_error = y_true[seasonality:] - y_true[:-seasonality]  
    naive_mae = mean_absolute_error(y_true[seasonality:], naive_forecast_error)
    mae = mean_absolute_error(y_true, y_pred)
    return mae / naive_mae

def normalized_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Normalized Absolute Error """
    mae = mean_absolute_error(y_true, y_pred)
    error = y_true - y_pred
    square_difference = np.square(error - mae)
    summed_square_difference = np.sum(square_difference)
    num_observations = len(y_true) - 1
    return np.sqrt(summed_square_difference / num_observations)

def normalized_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Normalized Absolute Percentage Error """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    percentage_error = (y_true - y_pred) / y_true
    square_difference = np.square(percentage_error - mape)
    summed_square_difference = np.sum(square_difference)
    num_observations = len(actual) - 1 
    return np.sqrt(summed_square_difference / num_observations)

def root_mean_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Root Mean Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    percentage_error = (y_true - y_pred) / y_true
    square_percentage_error = np.square(percentage_error)
    return np.sqrt(np.mean(
        square_percentage_error
    ))

def root_median_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Root Median Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    percentage_error = (y_true - y_pred) / y_true
    square_percentage_error = np.square(percentage_error)
    return np.sqrt(np.median(
        square_percentage_error
    ))

def root_mean_squared_scaled_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Root Mean Squared Scaled Error """
    benchmark = seasonal_decompose(y_true).seasonal.max()
    benchmark = math.floor(benchmark)
    if benchmark == 0:
        seasonality = 1
    else:
        seasonality = benchmark

    error = y_true - y_pred
    naive_forecast_error = y_true[seasonality:] - y_true[:-seasonality]
    mae = mean_absolute_error(actual[seasonality:], naive_forecast_error) 
    absolute_error_scaled = np.abs(error / mae)
    squared_absolute_error_scaled = np.square(absolute_error_mae_normalized)
    return np.sqrt(np.mean(
        squared_absolute_error_scaled
    ))


def integral_normalized_root_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Integral Normalized Root Squared Error """
    error = y_true - y_pred
    squared_error = np.square(error)
    normed_error = np.sqrt(np.sum(
        squared_error
    ))
    average_deviation = y_true - np.mean(y_true)
    squared_average_deviation = np.square(average_deviation)
    summed_squared_average_deviation = np.sum(
        squared_average_deviation
    )
    return normed_error / summed_suqared_average_deviation 


def rrse(y_true: np.ndarray, y_pred: np.ndarray):
    """ Root Relative Squared Error """
    error = y_true - y_pred
    squared_error = np.square(error)
    summed_squared_error = np.sum(squared_error)
    average_deviation = y_true - np.mean(y_true)
    squared_average_deviation = np.square(average_deviation)
    summed_squared_average_deviation = np.sum(
        squared_average_deviation
    )
    return np.sqrt(summed_squared_error / summed_squared_average_deviation)


def mre(y_true: np.ndarray, y_pred: np.ndarray):
    """ Mean Relative Error """
    return np.mean(_relative_error(y_true, y_pred))


def rae(y_true: np.ndarray, y_pred: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    absolute_error = np.abs(y_true - y_pred)
    summed_absolute_error = np.sum(absolute_error)
    average_deviation = y_true - np.mean(y_true)
    absolute_deviation = np.abs(average_deviation)
    summed_absolute_deviation = np.sum(
        absolute_deviation
    )
    return summed_absolute_error / summed_absolute_deviation


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Median Relative Absolute Error """
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def geometric_mean_relative_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Geometric Mean Absolute Percentage Error
    """
    return mstats.gmean(np.abs(
        _relative_error(y_true, y_pred, benchmark)
    ))
    
def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Geometric Mean Relative Absolute Error """
    return 


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Bounded Relative Absolute Error """
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Unscaled Mean Bounded Relative Absolute Error """
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

