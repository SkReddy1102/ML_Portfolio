# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 18:35:18 2021

@author: OKOK PROJECTS
"""

#import librarries required for CNN modeling
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator

filename = 'D:/PYTHON/RAGURAM/smart/bin.csv' #load CSV file
# from google.colab import files
# uploaded = files.upload()
# import io
# df = pd.read_csv(io.StringIO(uploaded['bin.csv'].decode('utf-8'))) # to read csv file from local directory
df = pd.read_csv(filename)
print(df.info())

df['eventDate'] = pd.to_datetime(df['eventDate']) #convert to datetime(time series'd)
df.set_axis(df['eventDate'], inplace=True) #making date as index
df.drop(columns=['id', 'type', 'Battery', 'updateState', 'Distance'], inplace=True) #drop columns not required for our modeling

df

#Plot FillPercentage Values
plt.figure(figsize=(16,8))
df['FillPercentage'].plot()
plt.xlabel('Date')
plt.ylabel('Fill Percentage')
plt.show()

fill_data = df['FillPercentage'].values
fill_data = fill_data.reshape((-1,1))

split_percent = 0.80 #split data as 80% training set and 20% test set
split = int(split_percent*len(fill_data))

fill_train = fill_data[:split]
fill_test = fill_data[split:]

date_train = df['eventDate'][:split]
date_test = df['eventDate'][split:]

print(len(fill_train))
print(len(fill_test))

#converting the data from sequence to supervised data to train CNN model

look_back = 15

train_generator = TimeseriesGenerator(fill_train, fill_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(fill_test, fill_test, length=look_back, batch_size=1)

#import keras models and layers for CNN
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense,RepeatVector
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(look_back,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam',metrics=["accuracy"]) #model compilation
model.summary() #model summary

num_epochs = 500
model.fit_generator(train_generator, epochs=num_epochs, verbose=1) #model training

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

testD = create_dataset(fill_test, 15)

import plotly.graph_objects as go #library for visualization

prediction = model.predict_generator(test_generator) #forecast/predict for test data

fill_train = fill_train.reshape((-1))
fill_test = fill_test.reshape((-1))
prediction = prediction.reshape((-1))

#Visualize Actual vs Predicted /Forecasted values
trace1 = go.Scatter(
    x = date_train,
    y = fill_train,
    mode = 'lines',
    name = 'Fill Percentage'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Forecasts'
)
trace3 = go.Scatter(
    x = date_test,
    y = fill_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "Waste Fill Levels",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Fill Percentage"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()
plt.close()

fill_data = fill_data.reshape((-1))

# Function to predict FillPercentage values for future dates
def predict(num_prediction, model):
    prediction_list = fill_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list

# Function to generate future dates    
def predict_dates(num_prediction):
    last_date = df['eventDate'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 30 #Forecast Future values for next 30 days i.e next one month.
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

# Plot values of future forecasted values
trace1 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Test Forecast'
)
trace2 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode='lines',
    name = 'Future Forecast'
)
layout = go.Layout(
    title = "Waste Fill Levels",
    xaxis = {'title' : "eventDate"},
    yaxis = {'title' : "Fill Percentage"}
)
fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()
plt.close()

# Plot for Actual vs Forecasted Value
plt.figure(figsize = (15,10))
plt.plot(fill_test)
plt.plot(prediction)
plt.title('Actual vs Prediction plot')
plt.ylabel('waste fill levels')
plt.xlabel('eventDate')
plt.legend(['actual', 'prediction'], loc='upper left')
plt.show()

fig, ax = plt.subplots(1, 2,figsize=(11,4))
ax[0].set_title('predicted one')
ax[0].plot(prediction)
ax[1].set_title('real one')
ax[1].plot(fill_test)
plt.show()

# Evaluation metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr})

forecast_accuracy(testD[1], prediction)



