import pandas as pd
from statsmodels.tsa import stattools
from statsmodels.stats import diagnostic
from statsmodels.tsa import arima_model
from collections import namedtuple
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from scipy.stats import mstats

def unscaled_mean_bounded_relative_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Unscaled Mean Bounded Relative Absolute Error
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
    Unscaled mean bounded relative absolute error as a float
    """
    numerator = [abs(elem - y_pred[idx]) for idx, elem in enumerate(y_true)]
    series_one = y_true[1:]
    series_two = y_true[:-1]
    denominator = [abs(elem - series_two[idx])
                   for idx, elem in enumerate(series_one)]
    final_series = [numerator[idx]/(numerator[idx] + denominator[idx])
                    for idx in range(len(denominator))]
    mbrae = np.mean(final_series)
    return mbrae/(1-mbrae)

def mean_bounded_relative_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Mean Bounded Relative Absolute Error
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
    Mean bounded relative absolute error as a float.
    """
    numerator = [abs(elem - y_pred[idx]) for idx, elem in enumerate(y_true)]
    series_one = y_true[1:]
    series_two = y_true[:-1]
    denominator = [abs(elem - series_two[idx]) for idx, elem in enumerate(series_one)]
    final_series = [numerator[idx]/(numerator[idx] + denominator[idx])
                    for idx in range(len(denominator))]
    return np.mean(final_series)

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
    percentage_error = (y_true - y_pred) / y_true)
    absolute_percentage_error = np.abs(percentage_error)
    return np.mean(absolute_percentage_error) * 100

def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
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

def mean_relative_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from: 
    http://www.spiderfinancial.com/support/documentation/numxl/reference-manual/forecasting-performance/mrae
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean Relative Absolute Error
    """
    numerator = [abs(elem - y_pred[idx])
                 for idx, elem in enumerate(y_true)]
    series_one = y_true[1:]
    series_two = y_true[:-1]
    denominator = [abs(elem - series_two[idx])
                   for idx, elem in enumerate(series_one)]    
    return np.mean([
        numerator[i]/denominator[i] for i in range(len(numerator))
    ])

def median_relative_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from: 
    http://www.spiderfinancial.com/support/documentation/numxl/reference-manual/forecasting-performance/mrae
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Median Relative Absolute Error
    """
    numerator = [abs(elem - y_pred[idx])
                 for idx, elem in enumerate(y_true)]
    series_one = y_true[1:]
    series_two = y_true[:-1]
    denominator = [abs(elem - series_two[idx])
                   for idx, elem in enumerate(series_one)]    
    return np.median([
        numerator[i]/denominator[i] for i in range(len(numerator))
    ])

def symmetric_mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
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
    numerator = [abs(y_pred[idx] - elem) for idx, elem in enumerate(y_true)]
    denominator = [abs(elem) + abs(y_pred[idx]) for idx, elem in enumerate(y_true)]
    denominator = [elem/2 for elem in denominator]
    result = np.mean([numerator[i]/denominator[i] for i in range(len(numerator))])
    return result * 100

def mean_absolute_scaled_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean Absolute scaled Error
    """
    numerator = sum([abs(y_pred[idx] - elem)  for idx, elem in enumerate(y_true)])
    series_one = y_true[1:]
    series_two = y_true[:-1]
    denominator = sum([abs(elem - series_two[idx])
                   for idx, elem in enumerate(series_one)])
    coeficient = len(y_true)/(len(y_true)-1)
    return numerator/(coeficient * denominator)

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
    numerator = [abs(y_pred[idx] - elem)  for idx, elem in enumerate(y_true)]
    series_one = y_true[1:]
    series_two = y_true[:-1]
    denominator = [abs(elem - series_two[idx])
                   for idx, elem in enumerate(series_one)]
    return mstats.gmean([numerator[i]/denominator[i] for i in range(len(numerator))])

def ad_fuller_test(timeseries):

    result = stattools.adfuller(timeseries)
    AdFullerResult = namedtuple('AdFullerResult', 'statistic pvalue')
    return AdFullerResult(result[0], result[1])

def kpss(timeseries):
    result = stattools.kpss(timeseries)
    KPSSResult = namedtuple('KPSSResult', 'statistic pvalue')
    return KPSSResult(result[0], result[1])

def cointegration(y_true, y_pred):
    result = stattools.coint(y_true, y_pred)
    CointegrationResult = namedtuple('CointegrationResult', 'statistic pvalue')
    return CointegrationResult(result[0], result[1])

def bds(timeseries):
    result = stattools.bds(timeseries)
    BdsResult = namedtuple('BdsResult', 'statistic pvalue')
    return BdsResult(result[0], result[1])

def q_stat(timeseries):
    autocorrelation_coefs = stattools.acf(timeseries)
    result = stattools.q_stat(autocorrelation_coefs)
    QstatResult = namedtuple('QstatResult', 'statistic pvalue')
    return QstatResult(result[0], result[1])

def acorr_ljungbox(timeseries):
    result = diagnostic.acorr_ljungbox(timeseries)
    AcorrLjungBoxResult = namedtuple('AcorrLjungBoxResult', 'statistic pvalue')
    return AcorrLjungBoxResult(result[0], result[1])

def acorr_breusch_godfrey(timeseries):
    result = diagnostic.acorr_breusch_godfrey(timeseries)
    AcorrBreuschGodfreyResult = namedtuple('BreuschGodfreyResult', 'statistic pvalue')
    return AcorrBreuschGodfreyResult(result[0], result[1])

def het_arch(timeseries):
    result = diagnostic.het_arch(timeseries)
    HetArchResult = namedtuple('HetArchResult', 'statistic pvalue')
    return HetArchResult(result[0], result[1])

def breaks_cumsumolsresid(timeseries):
    result = diagnostic.breaks_cusumolsresid(timeseries)
    BreaksCumSumResult = namedtuple('BreaksCumSumResult', 'statistic pvalue')
    return BreaksCumSumResult(result[0], result[1])
