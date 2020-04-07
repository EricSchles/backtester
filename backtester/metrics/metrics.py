import pandas as pd

def unscaled_mean_bounded_relative_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Unscaled Mean Bounded Relative Absolute Error
    Formula taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.series 
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
    * y_pred : pd.series 
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
    * y_pred : pd.series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean absolute percentage error as a float.
    """
    percentage_error = (y_true - y_pred) / y_true)
    absolute_percentage_error = np.abs(percentage_error)
    return np.mean(absolute_percentage_error) * 100

