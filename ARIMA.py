import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import log
from statsmodels.tsa.arima.model import ARIMA

parser = lambda date: pd.datetime.strptime(date, '%Y%b%d')
data_patient1 = pd.read_csv("shifted.csv", parse_dates=['day'], index_col=['day'])

mood = data_patient1[['mood']]
mood = mood.dropna()

mood_diff = mood.diff()

plt.plot(mood_diff)
mood_diff.dropna(inplace=True)
#plt.show()


# Dickey–Fuller test:
result = adfuller(mood_diff['mood'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))



# plot PACF to get ARIMA order


plot_pacf(mood_diff.dropna()) # only lag 1 is significant, therfore p=1
plot_acf(mood_diff.dropna())  # only lag 1 is significant, therefore q=1

#plt.show()

# build ARIMA model, order=(1,1,1)
model = ARIMA(mood, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# look at residuals
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])

# actual vs fitted
print(model_fit.forecast(steps=1))
#plt.show()

