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
        metric(y_true, y_pred)
        return True
    except:
        return False

def test__relative_error():
    assert example_code(bt_metrics._relative_error)

def test__bounded_relative_error():
    assert example_code(bt_metrics._bounded_relative_error)

def test_root_mean_squared_error():
    assert example_code(bt_metrics.root_mean_squared_error)
    
