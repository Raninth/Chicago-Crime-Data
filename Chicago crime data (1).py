#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
#import tick customization tools
import matplotlib.ticker as mticks


# In[7]:


get_ipython().system('pip')


# In[12]:


file_path = "Crimes_-_2001_to_Present.csv"
data = pd.read_csv(file_path)


# In[14]:


data.head()


# In[16]:


data['Date'] = pd.to_datetime(data['Date'])


# In[17]:


data.info()


# **1) Comparing Police Districts**
# - Which district had the most crimes in 2022?
# - Which had the least?
# 

# In[20]:


#Which district had the most crimes in 2022?
# Filter data for the year 2022
crimes_2022 = data[data['Year'] == 2022]


# In[22]:


district_crime_counts = crimes_2022['District'].value_counts()
district_crime_counts


# In[23]:


# District with the most crimes in 2022
district_most_crimes = district_crime_counts.idxmax()
most_crimes_count = district_crime_counts.max()

# District with the least crimes in 2022
district_least_crimes = district_crime_counts.idxmin()
least_crimes_count = district_crime_counts.min()


# In[24]:


district_least_crimes


# In[25]:


least_crimes_count


# In[26]:


district_most_crimes


# In[27]:


most_crimes_count


# In[28]:


print(f"District with the most crimes in 2022: {district_most_crimes} with {most_crimes_count} crimes")
print(f"District with the least crimes in 2022: {district_least_crimes} with {least_crimes_count} crimes")


# In[29]:


sorted_districts = district_crime_counts.sort_values(ascending=False)

# Plotting the top districts with the most crimes
plt.figure(figsize=(10, 6))
sorted_districts.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Districts with the Most Crimes')
plt.xlabel('District')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[30]:


sorted_districts = district_crime_counts.sort_values(ascending=True)

# Plotting the districts with the lowest crimes
plt.figure(figsize=(10, 6))
sorted_districts.head(10).plot(kind='bar', color='lightgreen')
plt.title('Districts with the lowest Crimes')
plt.xlabel('District')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# **2) Comparing Months:**
# - What months have the most crime? What months have the least?
# - Are there any individual crimes that do not follow this pattern? If so, which crimes?
# 

# In[33]:


data['Month'] = data['Date'].dt.month


# In[35]:


monthly_crime_counts = data['Month'].value_counts().sort_index().reset_index()
monthly_crime_counts.columns = ['Month', 'Crime Count']


# In[36]:


monthly_crime_counts


# In[37]:


month_most_crime = monthly_crime_counts[monthly_crime_counts['Crime Count'] == monthly_crime_counts['Crime Count'].max()]
month_least_crime = monthly_crime_counts[monthly_crime_counts['Crime Count'] == monthly_crime_counts['Crime Count'].min()]


# In[38]:


print("Month(s) with the most crimes:")
print(month_most_crime)

print("\nMonth(s) with the least crimes:")
print(month_least_crime)


# In[39]:


plt.figure(figsize=(10, 6))
plt.bar(monthly_crime_counts['Month'], monthly_crime_counts['Crime Count'], color='skyblue')
plt.xlabel('Month')
plt.ylabel('Crime Count')
plt.title('Crime Counts per Month')
plt.xticks(range(1, 13))  # Setting x-axis ticks for months (1 to 12)
plt.grid(axis='y')
plt.show()


# In[42]:


crime_counts_by_type_month = data.groupby(['Primary Type', 'Month']).size().reset_index(name='Count')
crime_anomalies = crime_counts_by_type_month[crime_counts_by_type_month['Month'].isin(monthly_crime_counts['Month']) & 
                                            (crime_counts_by_type_month['Count'] < crime_counts_by_type_month['Count'].quantile(0.25))]


# In[43]:


print("\nCrimes that do not follow the general pattern:")
print(crime_anomalies)


# In[44]:


plt.figure(figsize=(10, 6))
for crime_type in crime_anomalies['Primary Type'].unique():
    subset = crime_anomalies[crime_anomalies['Primary Type'] == crime_type]
    plt.plot(subset['Month'], subset['Count'], marker='o', label=crime_type)

plt.xlabel('Month')
plt.ylabel('Crime Count')
plt.title('Anomalies: Crimes with Lower Counts in High-Crime Months')
plt.xticks(range(1, 13))  # Setting x-axis ticks for months (1 to 12)
plt.legend()
plt.grid()
plt.show()


# **Topic 3) Crimes Across the Years:**
# 
# - Is the total number of crimes increasing or decreasing across the years?
# - Are there any individual crimes that are doing the opposite (e.g., decreasing when overall crime is increasing or vice-versa)?

# In[47]:


data['Year'] = data['Date'].dt.year

# Group data by year and count the number of crimes
crime_counts_by_year = data['Year'].value_counts().sort_index().reset_index()
crime_counts_by_year.columns = ['Year', 'Crime Count']


# In[48]:


crime_counts_by_year


# In[50]:


plt.figure(figsize=(10, 6))
plt.plot(crime_counts_by_year['Year'], crime_counts_by_year['Crime Count'], marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Total Crime Count')
plt.title('Total Crimes Across Years')
plt.grid()
plt.show()


# **Is the total number of crimes increasing or decreasing across the years?
# - it is decreasing

# In[52]:


overall_trend = 'Increasing' if crime_counts_by_year['Crime Count'].iloc[-1] > crime_counts_by_year['Crime Count'].iloc[0] else 'Decreasing'

# Function
def get_individual_crime_trends(data):
    individual_crime_trends = {}
    for crime_type, crime_type_data in data.groupby('Primary Type'):
        crime_counts_by_year_type = crime_type_data['Year'].value_counts().sort_index().reset_index()
        crime_counts_by_year_type.columns = ['Year', 'Crime Count']
        trend = 'Increasing' if crime_counts_by_year_type['Crime Count'].iloc[-1] > crime_counts_by_year_type['Crime Count'].iloc[0] else 'Decreasing'
        if trend != overall_trend:
            individual_crime_trends[crime_type] = trend
    return individual_crime_trends


individual_crime_trends = get_individual_crime_trends(data)

# Displaying overall trend
print(f"Overall trend of total crimes: {overall_trend}")
print("\nIndividual crimes with opposite trends:")
for crime, trend in individual_crime_trends.items():
    print(f"{crime}: {trend}")


# **Part 2**

# In[54]:


crime_types = ['THEFT', 'BATTERY', 'NARCOTICS', 'ASSAULT']


# In[56]:


crime_data = data[data['Primary Type'].isin(crime_types)]


# In[57]:


crime_data


# In[ ]:





# In[58]:


crime_per_month = crime_data.groupby(['Primary Type', pd.Grouper(key='Date', freq='M')]).size().reset_index(name='Crime Count')


# In[59]:


crime_per_month


# In[60]:


crime_per_month.isna().sum()


# In[61]:


pip install pmdarima


# In[62]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as            sns
import os
import statsmodels.tsa.api as tsa
from pmdarima.model_selection import train_test_split
from pmdarima.arima.utils import ndiffs, nsdiffs

# Set wide fig size for plots
plt.rcParams['figure.figsize']=(12,3)


# In[63]:


def plot_forecast(ts_train, ts_test, forecast_df, n_train_lags=None,
                  figsize=(10,4), title='Comparing Forecast vs. True Data'):
    ### PLot training data, and forecast (with upper/,lower ci)
    fig, ax = plt.subplots(figsize=figsize)

    # setting the number of train lags to plot if not specified
    if n_train_lags==None:
        n_train_lags = len(ts_train)

    # Plotting Training  and test data
    ts_train.iloc[-n_train_lags:].plot(ax=ax, label="train")
    ts_test.plot(label="test", ax=ax)

    # Plot forecast
    forecast_df['mean'].plot(ax=ax, color='green', label="forecast")

    # Add the shaded confidence interval
    ax.fill_between(forecast_df.index,
                    forecast_df['mean_ci_lower'],
                   forecast_df['mean_ci_upper'],
                   color='green', alpha=0.3,  lw=2)

    # set the title and add legend
    ax.set_title(title)
    ax.legend();

    return fig, ax



# In[64]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def regression_metrics_ts(ts_true, ts_pred, label="", verbose=True, output_dict=False,):
    # Get metrics
    mae = mean_absolute_error(ts_true, ts_pred)
    mse = mean_squared_error(ts_true, ts_pred)
    rmse = mean_squared_error(ts_true, ts_pred, squared=False)
    r_squared = r2_score(ts_true, ts_pred)
    mae_perc = mean_absolute_percentage_error(ts_true, ts_pred) * 100

    if verbose == True:
        # Print Result with label
        header = "---" * 20
        print(header, f"Regression Metrics: {label}", header, sep="\n")
        print(f"- MAE = {mae:,.3f}")
        print(f"- MSE = {mse:,.3f}")
        print(f"- RMSE = {rmse:,.3f}")
        print(f"- R^2 = {r_squared:,.3f}")
        print(f"- MAPE = {mae_perc:,.2f}%")

    if output_dict == True:
        metrics = {
            "Label": label,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R^2": r_squared,
            "MAPE(%)": mae_perc,
        }
        return metrics



# In[ ]:


# Custom function for Ad Fuller Test
def get_adfuller_results(ts, alpha=.05, label='adfuller', **kwargs): #kwargs for adfuller()
    # Saving each output
    (test_stat, pval, nlags, nobs, crit_vals_d,
    icbest ) = tsa.adfuller(ts, **kwargs)
    # Converting output to a dictionary with the interpretation of p
    adfuller_results = {'Test Statistic': test_stat,
                        "# of Lags Used":nlags,
                       '# of Observations':nobs,
                        'p-value': round(pval,6),
                        'alpha': alpha,
                       'sig/stationary?': pval < alpha}
    return pd.DataFrame(adfuller_results, index =[label])



# In[65]:


def plot_acf_pacf(ts, nlags=40, figsize=(10, 5),
                  annotate_sig=False, alpha=.05,
                 acf_kws={}, pacf_kws={},
                  annotate_seas=False, m = None,
                 seas_color='black'):

    fig, axes = plt.subplots(nrows=2, figsize=figsize)


    # Sig lags line style
    sig_vline_kwargs = dict( ls=':', lw=1, zorder=0, color='red')

    # ACF
    tsa.graphics.plot_acf(ts, ax=axes[0], lags=nlags, **acf_kws)

    ## Annotating sig acf lags
    if annotate_sig == True:
        sig_acf_lags = get_sig_lags(ts,nlags=nlags,alpha=alpha, type='ACF')
        for lag in sig_acf_lags:
            axes[0].axvline(lag,label='sig', **sig_vline_kwargs )

    # PACF
    tsa.graphics.plot_pacf(ts,ax=axes[1], lags=nlags, **pacf_kws)

    ## Annotating sig pacf lags
    if annotate_sig == True:
        ## ANNOTATING SIG LAGS
        sig_pacf_lags = get_sig_lags(ts,nlags=nlags,alpha=alpha, type='PACF')
        for lag in sig_pacf_lags:
            axes[1].axvline(lag, label='sig', **sig_vline_kwargs)




    ### ANNOTATE SEASONS
    if annotate_seas == True:
        # Ensure m was defined
        if m is None:
            raise Exception("Must define value of m if annotate_seas=True.")

        ## Calculate number of complete seasons to annotate
        n_seasons = nlags//m

        # Seasonal Lines style
        seas_vline_kwargs = dict( ls='--',lw=1, alpha=.7, color=seas_color, zorder=-1)

        ## for each season, add a line
        for i in range(1, n_seasons+1):
            axes[0].axvline(m*i, **seas_vline_kwargs, label="season")
            axes[1].axvline(m*i, **seas_vline_kwargs, label="season")

    fig.tight_layout()

    return fig



# In[71]:


crime_per_month.plot();


# In[68]:


crime_per_month.set_index('Date', inplace=True)


# In[69]:


crime_per_month


# In[73]:


crime_per_month = crime_per_month.resample('M').size()
crime_per_month


# In[74]:


# Apply seasonal decomposition
decomp = tsa.seasonal_decompose(crime_per_month)
fig = decomp.plot()
fig.set_size_inches(12,5)
fig.tight_layout()



# In[ ]:





# In[77]:


# Custom function for Ad Fuller Test
def get_adfuller_results(ts, alpha=.05, label='adfuller', **kwargs): #kwargs for adfuller()
    # Saving each output
    (test_stat, pval, nlags, nobs, crit_vals_d,
    icbest ) = tsa.adfuller(ts, **kwargs)
    # Converting output to a dictionary with the interpretation of p
    adfuller_results = {'Test Statistic': test_stat,
                        "# of Lags Used":nlags,
                       '# of Observations':nobs,
                        'p-value': round(pval,6),
                        'alpha': alpha,
                       'sig/stationary?': pval < alpha}
    return pd.DataFrame(adfuller_results, index =[label])



# In[78]:


#Determine if nonseasonal and/or seasonal differencing is required

get_adfuller_results(crime_per_month)


# In[83]:


# Determine differencing
d = ndiffs(crime_per_month)
print(f'd is {d}')
D = nsdiffs(crime_per_month, m = 12)
print(f'D is {D}')


# In[82]:


decomp.seasonal.loc["2018":"2022"].plot(marker = 'o')


# m = 12

# In[84]:


ts_diff = crime_per_month.diff().dropna()
ts_diff.plot();


# In[88]:


#Determine if nonseasonal and/or seasonal differencing is required

plot_acf_pacf(ts_diff, annotate_seas=True, m=6, nlags=20)


# In[89]:


#Split the time series into training and test data (Remember we want to predict 6 months)

train_size = len(crime_per_month) - 6  # Training set includes all data except the last 6 months
train, test = train_test_split(crime_per_month, train_size=train_size)

# Visualize train-test-split
ax = train.plot(label="train")
test.plot(label="test", ax=ax)
ax.legend()
plt.show()


# - fit a manual ARIMA/SARIMA model based on the orders determined during your exploration.
# - Make forecasts with your model.
# - Plot the forecasts versus the test data
# - Obtain metrics for evaluation

# In[90]:


# Orders for non seasonal components
p = 0 # nonseasonal AR
d = 1 # nonseasonal differencing
q = 2 # nonseasonal MA

# Orders for seasonal components (if seasonal model)
P = 1  # Seasonal AR
D = 0  # Seasonal differencing
Q = 1  # Seasonal MA
m = 6 # Seasonal period

sarima = tsa.ARIMA(train, order = (p,d,q), seasonal_order=(P,D,Q,m)).fit()


# In[91]:


sarima.summary()


# In[92]:


# Obtain diagnostic plots
fig = sarima.plot_diagnostics()
fig.set_size_inches(10,6)
fig.tight_layout()


# In[94]:


# Obtain summary of forecast as dataframe
forecast_df = sarima.get_forecast(len(test)).summary_frame()
forecast_df


# In[96]:


# Plot the forecast with true values
plot_forecast(train, test, forecast_df, n_train_lags = 20);


# In[97]:


regression_metrics_ts(test, forecast_df['mean'])


# - Tune with pmdarima's auto_arima
# - Fit a model on training data with the best parameters from auto_arima
# - Obtain metrics for evaluation
# - Make forecasts with the auto_arima model
# - Plot the forecasts versus the test data

# In[98]:


import pmdarima as pm
# Default auto_arima will select model based on AIC score
auto_model = pm.auto_arima(
    train,
    seasonal=True,  # True or False
    m=6,  # if seasonal
    trace=True)


# In[102]:


final_p = 0
final_q = 1
final_d = 2
final_P = 1
final_Q = 0
final_D = 1
m = 6
second_model = tsa.ARIMA(
    train,
    order=(final_p, final_d, final_q),
    seasonal_order=(final_P, final_D, final_Q, m),
).fit()


# In[103]:


second_model


# In[107]:


# Plot the forecast with true values
# Ger forecast into true future (fit on entrie time series)
forecast_df = second_model.get_forecast(len(test)).summary_frame()

plot_forecast(train, test, forecast_df, n_train_lags =50);


# In[108]:


regression_metrics_ts(test, forecast_df['mean'])


# In[109]:


#final model


# In[116]:


final_p = 0
final_q = 1
final_d = 2
final_P = 1
final_Q = 0
final_D = 1
m = 6
final_model = tsa.ARIMA(
    crime_per_month,
    order=(final_p, final_d, final_q),
    seasonal_order=(final_P, final_D, final_Q, m),
).fit()


# In[118]:


# Ger forecast into true future (fit on entrie time series)
forecast_df = final_model.get_forecast(len(test)).summary_frame()

plot_forecast(train, test, forecast_df, n_train_lags =50);



# In[119]:


# Define starting and final values
starting_value = forecast_df['mean'].iloc[0]
final_value = forecast_df['mean'].iloc[-1]
# Change in x
delta = final_value - starting_value
print(f'The change in X over the forecast is {delta: .2f}.')
perc_change = (delta/starting_value) *100
print (f'The percentage change is {perc_change :.2f}%.')



# In[120]:


#NEW
#Theft
#Battery


# In[121]:


# Filter data for 'THEFT' incidents
theft_data = crime_data[crime_data['Primary Type'] == 'THEFT']

# Group data by year-month to get the crime count per month for theft
theft_counts_per_month = theft_data.groupby(pd.Grouper(key='Date', freq='M')).size()

# Displaying the time series of crime counts per month for theft
print(theft_counts_per_month)


# In[122]:


theft_counts_per_month.isna().sum()


# In[123]:


# Apply seasonal decomposition
decomp = tsa.seasonal_decompose(theft_counts_per_month)
fig = decomp.plot()
fig.set_size_inches(12,5)
fig.tight_layout()


# In[124]:


# zooming in on smaller time period to see length of season
decomp.seasonal.loc["2018":"2022"].plot(marker = 'o')


# m = 12

# In[125]:


get_adfuller_results(theft_counts_per_month)


# In[127]:


# Determine differencing
d = ndiffs(theft_counts_per_month)
print(f'd is {d}')
D = nsdiffs(theft_counts_per_month, m = 12)
print(f'D is {D}')


# In[129]:


ts_diff = theft_counts_per_month.diff().dropna()
ts_diff.plot();


# In[131]:


plot_acf_pacf(ts_diff, annotate_seas=True, m=12, nlags=50)


# In[132]:


total_months = len(theft_counts_per_month)

# Calculate the number of months for the test set (6 months to predict)
months_to_predict = 6

# Calculate the test size as a proportion of the total data
test_size = months_to_predict / total_months


# In[133]:


test_size


# In[151]:


from pmdarima.model_selection import train_test_split
train, test = train_test_split(theft_counts_per_month, test_size= 0.021)
# Visualize train-test-split
ax = train.plot(label="train")
test.plot(label="test")
ax.legend();


# In[152]:


# Orders for non seasonal components
p = 1 # nonseasonal AR
d = 1 # nonseasonal differencing
q = 1 # nonseasonal MA

# Orders for seasonal components (if seasonal model)
P = 1  # Seasonal AR
D = 0  # Seasonal differencing
Q = 1  # Seasonal MA
m = 12 # Seasonal period

sarima = tsa.ARIMA(train, order = (p,d,q), seasonal_order=(P,D,Q,m)).fit()


# In[153]:


# Obtain summary of forecast as dataframe
forecast_df = sarima.get_forecast(len(test)).summary_frame()
# Plot the forecast with true values
plot_forecast(train, test, forecast_df, n_train_lags = 20)


# In[154]:


regression_metrics_ts(test, forecast_df['mean'])


# In[141]:


import pmdarima as pm
# Default auto_arima will select model based on AIC score
auto_model = pm.auto_arima(
    train,
    seasonal=True,  # True or False
    m=12,  # if seasonal
    trace=True)


# In[143]:


final_p = 0
final_q = 1
final_d = 0
final_P = 1
final_Q = 0
final_D = 1

final_model = tsa.ARIMA(
    train,
    order=(final_p, final_d, final_q),
    seasonal_order=(final_P, final_D, final_Q, m),
).fit()



# In[144]:


# Ger forecast into true future (fit on entrie time series)
forecast_df = final_model.get_forecast(len(test)).summary_frame()

plot_forecast(train, test, forecast_df, n_train_lags =20);



# In[145]:


regression_metrics_ts(test, forecast_df['mean'])


# In[146]:


final_p = 0
final_q = 1
final_d = 0
final_P = 1
final_Q = 0
final_D = 1

final_model = tsa.ARIMA(
    theft_counts_per_month
,
    order=(final_p, final_d, final_q),
    seasonal_order=(final_P, final_D, final_Q, m),
).fit()




# In[147]:


# Ger forecast into true future (fit on entrie time series)
forecast_df = final_model.get_forecast(len(test)).summary_frame()

plot_forecast(train, test, forecast_df, n_train_lags =20);



# In[155]:


# Define starting and final values
starting_value = forecast_df['mean'].iloc[0]
final_value = forecast_df['mean'].iloc[-1]
# Change in x
delta = final_value - starting_value
print(f'The change in X over the forecast is {delta: .2f}.')
perc_change = (delta/starting_value) *100
print (f'The percentage change is {perc_change :.2f}%.')



# In[ ]:




