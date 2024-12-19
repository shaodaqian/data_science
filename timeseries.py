import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller,kpss

from datetime import datetime
import requests
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sns


air2 = requests.get('https://www.stata-press.com/data/r12/air2.dta').content
data = pd.read_stata(BytesIO(air2))
data.index = pd.date_range(start=datetime(int(data.time[0]), 1, 1), periods=len(data), freq='MS')
data['lnair'] = np.log(data['air'])
data['D.lnair'] = data['lnair'].diff()

test_res=adfuller(data['D.lnair'][1:],regression='c')
print(test_res)
print(data.head())
print(data.info())
print(data.describe())

data['air'].resample('Q').agg(['mean', 'std']).plot(subplots=True, figsize=(15, 6))
plt.show()


data['air'].plot(label='air')
data['air'].rolling(6).mean().plot(label='rolling 6')
data['air'].rolling(6,center=True).mean().plot(label='rolling 6 center')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(15,4))

# Levels
axes[0].plot(data.index._mpl_repr(), data['air'], '-')
axes[0].set(title='US Wholesale Price Index')

# Log difference
axes[1].plot(data.index._mpl_repr(), data['D.lnair'], '-')
axes[1].hlines(0, data.index[0], data.index[-1], 'r')
axes[1].set(title='log diff');


fig, axes = plt.subplots(1, 2, figsize=(15,4))
fig = sm.graphics.tsa.plot_acf(data.iloc[1:]['D.lnair'], lags=40, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(data.iloc[1:]['D.lnair'], lags=40, ax=axes[1])

plt.show()

train_ts=data['lnair'][:'1959-01-01']

# Fit the model
train_model = sm.tsa.statespace.SARIMAX(train_ts, order=(2,1,0), seasonal_order=(1,1,0,12))
res = train_model.fit(disp=False)
print(res.summary())


val_model = sm.tsa.statespace.SARIMAX(data['lnair'], order=(2,1,0), seasonal_order=(1,1,0,12))

res = val_model.filter(res.params)
predict = res.get_prediction()
predict_ci = predict.conf_int()
print(predict.predicted_mean,'val')

predict_dy = res.get_prediction(end='1962', dynamic='1959-01-01')
predict_dy_ci = predict_dy.conf_int()
print(predict_dy.predicted_mean,'dynamic')

# Graph
fig, ax = plt.subplots(figsize=(9,4))
ax.set(title='Personal consumption', xlabel='Date', ylabel='Billions of dollars')

# Plot data points
data.loc['1949-01-01':, 'lnair'].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.loc['1951-01-01':'1959-01-01'].plot(ax=ax, style='r--', label='in sample One-step-ahead')
ci = predict_ci.loc['1951-01-01':'1959-01-01']
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

predict.predicted_mean.loc['1959-01-01':].plot(ax=ax, style='b--', label='val One-step-ahead')
ci = predict_ci.loc['1959-01-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='b', alpha=0.1)

predict_dy.predicted_mean.loc['1959-01-01':].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
ci = predict_dy_ci.loc['1959-01-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

legend = ax.legend(loc='lower right')

plt.show()


fig, ax = plt.subplots(figsize=(9,4))
ax.set(title='Forecast error', xlabel='Date', ylabel='Forecast - Actual')

# In-sample one-step-ahead predictions and 95% confidence intervals
predict_error = predict.predicted_mean - data['lnair']
predict_error.loc['1951-01-01':'1959-01-01'].plot(ax=ax, label='in sample One-step-ahead')
ci = predict_ci.loc['1951-01-01':'1959-01-01'].copy()
ci.iloc[:,0] -= data['lnair'].loc['1951-01-01':'1959-01-01']
ci.iloc[:,1] -= data['lnair'].loc['1951-01-01':'1959-01-01']
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.1)


predict_error = predict.predicted_mean - data['lnair']
predict_error.loc['1959-01-01':].plot(ax=ax, label='val One-step-ahead')
ci = predict_ci.loc['1959-01-01':].copy()
ci.iloc[:,0] -= data['lnair'].loc['1959-01-01':]
ci.iloc[:,1] -= data['lnair'].loc['1959-01-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.1)



# Dynamic predictions and 95% confidence intervals
predict_dy_error = predict_dy.predicted_mean - data['lnair']
predict_dy_error.loc['1959-01-01':].plot(ax=ax, style='r', label='Dynamic forecast (1978)')
ci = predict_dy_ci.loc['1959-01-01':].copy()
ci.iloc[:,0] -= data['lnair'].loc['1959-01-01':]
ci.iloc[:,1] -= data['lnair'].loc['1959-01-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

legend = ax.legend(loc='lower left');
legend.get_frame().set_facecolor('w')

plt.show()