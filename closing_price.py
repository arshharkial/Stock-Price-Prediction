#Importind necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import h5py

#Importing dataset
dataset = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
dataset.head()

#Converting Date from string to Python Date Time
dataset.Date = pd.to_datetime(dataset.Date, format = "%Y-%m-%d")
dataset.index = dataset.Date
dataset = dataset.sort_index(ascending = True, axis = 0)

#Plotting closing points
plt.plot(dataset.Close, label = "Close Price History")

#Create new_dataset from Date and Close columns
new_dataset = dataset[['Date', 'Close']].copy()

#Normalise data
train_size = 0.8
split = len(dataset) * train_size
scaler = MinMaxScaler()

train_data = new_dataset.iloc[0 : int(split), : ].values
test_data = new_dataset.iloc[int(split) : , : ].values

scaled_dataset = scaler.fit_transform(new_dataset.iloc[:, 1].values.reshape(-1, 1))

new_dataset.drop("Date", axis = 1, inplace = True)

x_train,y_train = [], []
for i in range(60,len(train_data)):
    x_train.append(scaled_dataset[i-60 : i, 0])
    y_train.append(scaled_dataset[i , 0])

x_train, y_train = np.array(x_train) , np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Create model
model = Sequential([
                    LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)),
                    LSTM(units = 50),
                    Dense(1)
                  ])

model.compile(loss = "mean_squared_error", optimizer = "adam")
model.fit(x_train, y_train, epochs = 9, batch_size = 8, verbose = 2)

#Create dataset to make prediction using lstm model
inputs_data = new_dataset[len(new_dataset) - len(test_data) - 60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#Create prediction
closing_price_prediction = model.predict(X_test)
closing_price_prediction = scaler.inverse_transform(closing_price_prediction)

#Plot Predicted stock costs vs actual stock costs
train_data=new_dataset[ : int(split)]
test_data=new_dataset[int(split) : ]
test_data['Predictions'] = closing_price_prediction
plt.plot(train_data["Close"])
plt.plot(test_data[['Close',"Predictions"]])