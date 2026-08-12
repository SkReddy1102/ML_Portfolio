# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 18:43:00 2021

@author: OKOK PROJECTS
"""

import csv
import json
from datetime import datetime
from dateutil import parser


fw_access_layers_data = open('D:/PYTHON/RAGURAM/smart/out_bins.json', 'r')
fw_access_layers_parsed = json.loads(fw_access_layers_data.read())
with open("bin.csv", "w+") as outfile:
    f = csv.writer(outfile)
    f.writerow(["id", "type", "FillPercentage", "Battery", "eventDate", "updateState","Distance"])
    for i in range(16801):
      dt =  fw_access_layers_parsed['data'][i]['request']['eventDate']
      date_object = parser.parse(dt)
      fw_access_layers_parsed['data'][i]['request']['eventDate'] = date_object.date()
      f.writerow([fw_access_layers_parsed['data'][i]['id'], fw_access_layers_parsed['data'][i]['type'],
                  fw_access_layers_parsed['data'][i]['request']['measurements']['FillPercentage'],
                  fw_access_layers_parsed['data'][i]['request']['measurements']['battery'],
                  fw_access_layers_parsed['data'][i]['request']['eventDate'],
                  fw_access_layers_parsed['data'][i]['request']['updateState'],
                  fw_access_layers_parsed['data'][i]['Distance']])
      
import pandas as pd
import numpy as np
from IPython.display import display
import datetime, pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

# from google.colab import files
# uploaded = files.upload()
# import io
# data= pd.read_csv(io.StringIO(uploaded['bin.csv'].decode('utf-8'))) # to read csv file from local directory
data =pd.read_csv('D:/PYTHON/RAGURAM/smart/bin.csv' ) #csv file generated is read using pandas library
data.head()


data.info() #summary of the data

data.isnull().sum() #data is checked for null or missing values

data['eventDate'] = pd.to_datetime(data['eventDate']) #converting the eventDate attribute of the dataset datetime to use in time series forecasting.

df = data.set_index('eventDate') #dataframe by "eventDate" as data index, i.e indexed by date

from statsmodels.tsa.stattools import adfuller

#Perform Augmented Dickey–Fuller test:
print('Results of Dickey Fuller Test:')
dftest = adfuller(df['FillPercentage'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)

plt.style.use('fivethirtyeight')
data.groupby('eventDate')['FillPercentage'].mean().plot(figsize=(18, 5), color='grey')

plt.style.use('fivethirtyeight')
data.groupby('eventDate')['Battery'].mean().plot(figsize=(22, 4), color='grey')

plt.style.use('fivethirtyeight')
data.groupby('FillPercentage')['Battery'].mean().plot(figsize=(22, 4), color='grey')

plt.style.use('fivethirtyeight')
cg = data.groupby('eventDate', as_index=False)['FillPercentage'].mean()
plt.figure(figsize=(24, 5))
sns.barplot(data=cg, x='eventDate', y='FillPercentage', palette='gray')

plt.style.use('fivethirtyeight')
cg = data.groupby('FillPercentage', as_index=False)['Battery'].mean()
plt.figure(figsize=(24, 5))
sns.barplot(data=cg, x='FillPercentage', y='Battery', palette='gray')

data.FillPercentage.describe() #summary of "FillPercentage" attribute

plt.style.use('ggplot') #Plot Plots for FillPercentage values
plt.figure(figsize=(8, 5))
pd.plotting.lag_plot(data['FillPercentage'], lag=1)

sns.set()  #Plot Plots for FillPercentage values using seaborn style
plt.style.use('seaborn')
plt.figure(figsize=(8, 5))
pd.plotting.lag_plot(data['FillPercentage'], lag=2)

sns.set() #Plot for autocorrelation 
plt.style.use('ggplot')
plt.figure(figsize=(10, 8))
pd.plotting.autocorrelation_plot(data['FillPercentage'])

plt.figure(figsize=(15, 10)) #heatmap to check correlation of various data attributes
sns.heatmap(data.corr(), annot=True, cmap='gray')

data.groupby(['eventDate']).FillPercentage.mean().plot() #plot for mean Fillpercenatge value