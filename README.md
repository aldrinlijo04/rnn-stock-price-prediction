# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
1. The given problem is to predict the google stock price based on time.
2. For this we are provided with a dataset which contains features like Date, Opening Price, Highest Price, Lowest Price, Closing Price, Adjusted Closing, Price and Volume.
3. Based on the given features, develop a RNN model to predict the price of stocks in future.

## Design Steps

### Step 1: Import Necessary Packages
Import the required libraries and packages needed for data manipulation, model building, and evaluation.

### Step 2: Load the Dataset
Load the dataset containing the relevant information for the pricing model. Ensure the data is properly formatted and accessible for analysis.

### Step 3: Perform Data Preprocessing
Preprocess the dataset to handle missing values, scale features, encode categorical variables, and any other necessary data transformations to prepare it for model training.

### Step 4: Build and Fit the Learning Model
Construct the appropriate machine learning model for pricing prediction and train it using the preprocessed data. This involves selecting the model architecture, specifying hyperparameters, and fitting the model to the training data.

### Step 5: Make Predictions
Utilize the trained model to make predictions on new or unseen data. Evaluate the model's performance in generating accurate price predictions.

### Step 6: Evaluate Model Performance
Assess the error metrics or other relevant evaluation criteria to determine the effectiveness of the pricing model. Analyze the discrepancy between predicted and actual prices to understand the model's accuracy and potential areas for improvement.

## Program
#### Name: Aldrin Lijo J E
#### Register Number: 212222240007
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values

type(train_set)

train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential()

model.add(layers.SimpleRNN(50, input_shape = (length, n_features)))
model.add(layers.Dense(1))

model.compile(optimizer = "adam", loss = "mse")

model.summary()

model.fit(X_train1,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('testset.csv')

test_set = dataset_test.iloc[:,1:2].values

test_set.shape

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
  y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

print("Name: Aldrin Lijo J E  Register Number: 212222240007")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```
## Output

### True Stock Price, Predicted Stock Price vs time
![download](https://github.com/aldrinlijo04/rnn-stock-price-prediction/assets/118544279/ba849922-1771-4d2b-b58a-dfe448140a9a)

### Mean Square Error
![image](https://github.com/aldrinlijo04/rnn-stock-price-prediction/assets/118544279/b257b663-867d-469e-a094-50af831aee92)

## Result
Thus, a Recurrent Neural Network model for stock price prediction is developed.
