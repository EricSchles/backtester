############
Introduction
############

Welcome to Backtester, a tool to help you back test your time series forecasting models.  This testing framework is broken out semantically, so you can analyze different parts of your time series forecast.

The framework comes in two general flavors, metrics to test accuracy of forecast like::

	from backtester import metrics as bt_metrics
	import pandas as pd
	import statsmodels.api as sm

	df = sm.datasets.sunspots.load_pandas().data
	df.index = pd.Index(
		sm.tsa.datetools.dates_from_range(
			'1700', '2008'
		)
	)
	train = df.iloc[:'2000']
	y_true = df.iloc['2000':]
	arma = sm.tsa.ARMA(train, (2, 0))
	arma.fit(disp=False)
	y_pred = arma.forecast(steps=8)
	print(bt_metrics.umbrae(y_true, y_pred))

