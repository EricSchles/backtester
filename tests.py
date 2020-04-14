from backtester import backtest
from backtester import metrics as bt_metrics
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import code

def example_code(metric):
    df = sm.datasets.sunspots.load_pandas().data
    df.index = pd.Index(
        sm.tsa.datetools.dates_from_range(
            '1700', '2008'
        )
    )
    df.drop("YEAR", axis=1, inplace=True)
    print(set(seasonal_decompose(
        df["SUNACTIVITY"], model="additive"
    ).seasonal))
    train = df.iloc[:-8]
    y_true = df.iloc[-8:]
    
    arma = sm.tsa.ARMA(train, (2, 0))
    res = arma.fit(disp=False)
    y_pred = res.forecast(steps=8)
    y_pred = pd.Series(y_pred[0])
    y_pred.index = y_true.index
    try:
        return metric(y_true, y_pred), True
    except:
        return False, False

def test__relative_error():
    result = example_code(bt_metrics._relative_error)
    assert result[1]
    
def test__bounded_relative_error():
    result = example_code(bt_metrics._bounded_relative_error)
    assert result[1]
    
def test_root_mean_squared_error():
    result = example_code(bt_metrics.root_mean_squared_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_normalized_root_mean_squared_error():
    result = example_code(bt_metrics.normalized_root_mean_squared_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_mean_error():
    result = example_code(bt_metrics.mean_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_absolute_error():
    result = example_code(bt_metrics.absolute_error)
    assert result[1]

def test_mean_absolute_error():
    result = example_code(bt_metrics.mean_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_median_absolute_error():
    result = example_code(bt_metrics.median_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_variance_absolute_error():
    result = example_code(bt_metrics.variance_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_iqr_absolute_error():
    result = example_code(bt_metrics.iqr_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_geometric_mean_absolute_error():
    result = example_code(bt_metrics.geometric_mean_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_mean_percentage_error():
    result = example_code(bt_metrics.mean_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_mean_absolute_percentage_error():
    result = example_code(bt_metrics.mean_absolute_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_median_absolute_percentage_error():
    result = example_code(bt_metrics.median_absolute_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_symmetric_mean_absolute_percentage_error():
    result = example_code(bt_metrics.symmetric_mean_absolute_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_symmetric_median_absolute_percentage_error():
    result = example_code(bt_metrics.symmetric_median_absolute_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_arctangent_absolute_percentage_error():
    result = example_code(bt_metrics.mean_arctangent_absolute_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_mean_absolute_scaled_error():
    result = example_code(bt_metrics.mean_absolute_scaled_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_normalized_absolute_error():
    result = example_code(bt_metrics.normalized_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_normalized_absolute_percentage_error():
    result = example_code(bt_metrics.normalized_absolute_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_root_mean_squared_percentage_error():
    result = example_code(bt_metrics.root_mean_squared_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_root_median_squared_percentage_error():
    result = example_code(bt_metrics.root_median_squared_percentage_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_root_mean_squared_scaled_error():
    result = example_code(bt_metrics.root_mean_squared_scaled_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_integral_normalized_root_squared_error():
    result = example_code(bt_metrics.integral_normalized_root_squared_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_root_relative_squared_error():
    result = example_code(bt_metrics.root_relative_squared_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_mean_relative_error():
    result = example_code(bt_metrics.mean_relative_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_relative_absolute_error():
    result = example_code(bt_metrics.relative_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_mean_relative_absolute_error():
    result = example_code(bt_metrics.mean_relative_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_median_relative_absolute_error():
    result = example_code(bt_metrics.median_relative_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_geometric_mean_relative_absolute_error():
    result = example_code(bt_metrics.geometric_mean_relative_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_mean_bounded_relative_absolute_error():
    result = example_code(bt_metrics.mean_bounded_relative_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_unscaled_mean_bounded_relative_absolute_error():
    result = example_code(bt_metrics.unscaled_mean_bounded_relative_absolute_error)
    assert isinstance(result[0], float)
    assert result[1]
    
def test_mean_directional_accuracy():
    result = example_code(bt_metrics.mean_directional_accuracy)
    assert isinstance(result[0], float)
    assert result[1]

