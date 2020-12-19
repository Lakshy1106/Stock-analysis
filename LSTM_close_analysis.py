#----------------------------------------------IMPORTING MODULES---------------------------------------
#Import the libraries
import tensorflow
import math
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#-----------------------------------------
# GATHERING INFORMATION
#-----------------------------------------
#Number of days to remember

c = int(input("Number of days for short term memory: "))
a=int(input ("Number of days for which analysis must be done: "))

# Getting the stock data
b = input("Enter the stock quote: ")
d= input("Enter the file extension in csv or xlsx: ")
address = "Database/{}.{}".format(b, d)
if d == 'csv':
	df = pd.read_csv(address, index_col="Date")
else:
	df = pd.read_excel(address, index_col="Date")

df = df.tail(c)
df.index = pd.to_datetime(df.index)

#-----------------------------------------
# DATA DISPLAY
#-----------------------------------------

#Visualize the closing price history
plt.figure(figsize=(14,7))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=16)
plt.ylabel('Close Price',fontsize=18)
plt.show()

#------------------------------------------
# DATA PROCESSING
#------------------------------------------

#Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
#Converting the dataframe to a numpy array
dataset = data.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8)

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]
#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(a,len(train_data)):
    x_train.append(train_data[i-a:i,0])
    y_train.append(train_data[i,0])

#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#-----------------------------------
# BUILDING THE MODEL
#-----------------------------------

#Build the LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='rmsprop', loss='mean_squared_error')

#---------------------------------
# TRAINING AND TESTING
#---------------------------------

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Test data set
test_data = scaled_data[training_data_len - a: , : ]

#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ]#Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(a,len(test_data)):
    x_test.append(test_data[i-a:i,0])

#Convert x_test to a numpy array 
x_test = np.array(x_test)

#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling

#----------------------------------------
# RESULTS
#----------------------------------------

#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print("The loss is given by ",rmse)

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualizing the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


#--------------------------
# APPLICATION FOR PREDICTION
#--------------------------

quote = pd.read_csv(address)
new_df = quote.filter(['Close'])
#Get the last days closing price and scale data 
last_days = new_df[-a:].values
last_days_scaled = scaler.transform(last_days)
#Create a empty list
X_test = []
X_test.append(last_days_scaled)
#Convert the test data set to a numpy array
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted price
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)

#--------------------------------
# END PREDICTION
#--------------------------------

print("\n---------------------------RESULTS ARE--------------------------------------------------")
print("\nThe predicted price for tomorrow based on previous {} days is - {} ".format(a,pred_price))
print("\n----------------------------CODE HAS ENDED------------------------------------------------")
