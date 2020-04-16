

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from datetime import datetime


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 15, 6


# # Initial Inferences

# In[3]:


data = pd.read_csv('device41.csv')
data.head(-10)


# In[4]:


data.dtypes


# In[5]:


dateparse = lambda dates: pd.datetime.strptime('2013-01-01 09:10:12','%Y-%m-%d %H:%M:%S')
data = pd.read_csv('device41.csv', parse_dates=['DateTime'], index_col='DateTime')
data.head()


# In[6]:


data.index


# In[7]:


ts = data['pm25']
ts.head(10)


# In[8]:


ts['2019-12-27']


# In[9]:


ts[datetime(2019,12,27)]


# In[10]:


ts['2019-12-27 00:00:00':'2019-12-27 23:59:40']


# In[11]:


ts['2019']


# In[12]:


plt.plot(ts)


# # Testing Stationarity

# Rolling Statistics (Visual Method) and Dickey Fuller Test (Statistical Method)

# In[13]:


from statsmodels.tsa.stattools import adfuller


# In[14]:


def rolling_stats(timeseries):
    #Window of 12 months
    rolmean = timeseries.rolling(10).mean()
    rolstd = timeseries.rolling(10).std()
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[15]:


def augmented_dickey_fuller(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput


# In[16]:


rolling_stats(ts)


# Inferences: Variation in Std. Deviation is less but Rolling Average increases with time. This means it is not a stationary series.

# In[17]:


augmented_dickey_fuller(ts)


# If the Test Statistic is less than the Critical Value, we can say that the series is stationary.

# Inferences: Test Statistic value way greater than the Critical values (Check signed values not absolute values)

# # Procedure Overview

# Time series models assume data is stationary which is rarely the case. So, we first have to make the series stationary. The factors making a time series non-stationary are:
# 
# 1. Trend: Irregular
# 2. Seasonality: Variation in values during days

# Procedure:
# 
# 1. Estimate Trend and Seasonality
# 2. Eliminate both and make the series stationary
# 3. Apply time series models on the stationary data to forecast future values
# 4. Restore the trend and seasonality constraints to the forecasted values to convert to the original scale

# # Trend Elimination by Model Estimation

# ## Transformation

# Trend Type: Increasing Average.
# We can use a transformation that penalize higher values more like log or sqrt

# In[18]:


ts_log = np.log(ts)
plt.plot(ts_log)


# The scale reduces, but the layout is still noisy. To estimate/model this trend and then remove it from the series would be a better approach. The possible methods are:
# 
# 1. Aggregation – Average for a time period like monthly/weekly averages
# 2. Smoothing – Rolling Average
# 3. Polynomial Fitting – Fit a regression model
# 
# We will try out the Smoothing method by calculating the rolling average.

# ## Moving Average

# In[19]:


moving_avg = ts_log.rolling(12).std()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


# In[20]:


ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)


# In[21]:


ts_log_moving_avg_diff.dropna(inplace=True)


# In[22]:


rolling_stats(ts_log_moving_avg_diff)


# In[23]:


augmented_dickey_fuller(ts_log_moving_avg_diff)


# This looks better but The test statistic is still greater than Critical value(5%)

# ## Exponential Weighted Moving Average

# Therefore, we use Exponentially Weighted Moving Average (EMWA) where weights are assigned to all the previous values with a decay factor. More emphasis is given to recent past values and importance decreases for the older values.

# In[24]:


expweighted_avg = pd.Series.ewm(ts_log, halflife=12).mean()
plt.plot(ts_log)
plt.legend(loc='best')
plt.plot(expweighted_avg, color='red')


# In[25]:


ts_log_ewma_diff = ts_log - expweighted_avg


# In[26]:


rolling_stats(ts_log_ewma_diff)


# In[27]:


augmented_dickey_fuller(ts_log_ewma_diff)


# This time, it can be said that the series is stationary with high confidence which is better than the previous case. All the values are assigned weights therefore there are no missing values.

# # Seasonality Elimination

# The simple trend reduction techniques used above do not work in all cases, especially when the seasonality is high. There are better methods to counter seasonality:
# 
# 1. Differencing: Taking the differece with a particular time lag
# 2. Decomposition: Modeling both Trend and Seasonality and removing them from the model.

# ## Differencing

# We take the difference of the observation at a particular instant with that at the previous instant. First order differencing:

# In[28]:


ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)


# In[29]:


ts_log_diff.dropna(inplace=True)


# In[30]:


rolling_stats(ts_log_diff)


# In[31]:


augmented_dickey_fuller(ts_log_diff)


# We can state the stationarity as good
# 

# ## Decomposing

# Both trend and seasonality are modelled separately and the remaining part of the series is returned. For more details watch these videos: <br/>
# 
# Seasonal Decomposition and Forecasting:
# 
# 1. https://www.youtube.com/watch?v=85XU1T9DIps (Part I)
# 2. https://www.youtube.com/watch?v=CfB9ROwF2ew (Part II)

# In[32]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[33]:


decomposition = seasonal_decompose(ts_log,freq=700)


# In[34]:


trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


# In[35]:


plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')


# In[36]:


plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')


# In[37]:


plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')


# In[38]:


plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# Now that the trend and seasonality has been removed, the residuals can be analysed for stationarity.

# In[39]:


ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)


# In[40]:


rolling_stats(ts_log_decompose)


# In[41]:


augmented_dickey_fuller(ts_log_decompose)


# 99% confidence for stationarity of the residuals

# # Forecasting

# Using the difference model to eliminate trend and seasonality as adding them back to the predicted residuals is relatively easier than other methods. We may be left with two situations:
# 
# 1. A strictly stationary series with no dependence among the values.
# 2. A series with significant dependence among the values. (Statistical models are required to forecast)

# We will be using the Auto Regressive Integrated Moving Averages (ARIMA) Model. The model is similiar to linear regression with the following parameters:
# 
# 1. Number of AR (Auto-Regressive) Terms (p)
# 2. Number of MA (Moving Average) terms (q)
# 3. Number of Differences (d)

# ## ACF and PACF

# In order to determine the values of p and q, we have to compute the following terms:
# 
# 1. Auto-correlation Function (ACF): Measure of co-relation of a time series with a lagged version of itself.
# 2. Partial Auto-correlation Function (PACF): ACF but after eliminating the variations already explained by previous successive lags.

# In[42]:


from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm


# In[43]:


lag_acf = acf(ts_log_diff, nlags=60)
lag_pacf = pacf(ts_log_diff, nlags=60, method='ols')


# In[44]:


plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Auto-correlation F6nction (ACF)')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Auto-correlation Function (PACF)')
plt.tight_layout()


# ## ARIMA Model

# We will prepare 3 different models considering the individual as well as the combined effects. <br/>
# 
# Note: RSS printed is for the residuals and not for the actual series

# In[45]:


from statsmodels.tsa.arima_model import ARIMA


# The parameters required for the ARIMA model can be filled in a tuple in the order (p,d,q)

# ### Case 1: AR Model

# In[ ]:


model = ARIMA(ts_log, order=(5, 1, 0))  
results_AR = model.fit(disp=5)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))


# ### Case 2: MA Model

# In[47]:


model = ARIMA(ts_log, order=(1, 1,0))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))


# # Restoring to Original Scale

# Continuing with the combined model, we scale it back to the original values. First, we store the predicted results as a separate series.

# In[49]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()


# In[50]:


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()


# In[51]:


predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# Last step is to take the exponent and compare with the original series.

# In[52]:


predictions_ARIMA = np.exp(predictions_ARIMA_log)


# In[53]:


plt.plot(ts, color='blue')
plt.plot(predictions_ARIMA, color='red')
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))


# In[ ]:


series = pd.read_csv('device41.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=dateparse)
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()



series = pd.read_csv('device41.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=dateparse)
autocorrelation_plot(series)







