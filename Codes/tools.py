# Importing all Libraries
#----------------------------
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import tensorflow
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from operator import add
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

# Defining all the tools
#-------------------------------------

def fibonacci_processing(data):
    # Reading last 'a' rows only.
    #-------------------------------------------------------------
    a = int(input ("\nEnter the number of days to be analysed : "))
    df = data.tail(a)
    df.index = pd.to_datetime(df.index)

    # Processing data
    #------------------------------------------------------------------
    min = df["Close"].min()
    max = df["Close"].max()
    price = df["Close"].last('1D')

    # Plotting Close prices graph for visual help.
    fig, ax = plt.subplots()
    ax.plot(df['Close'], color='black')
    plt.xlabel("Date")
    plt.title("Preview chart")
    plt.show()

    c = int(input("What is the trend type?\n(1- upward 0 - downward): "))
    #-------------------------------------------------------------------------------------------

    if (c==1):
        # Making Fibonacci Levels considering original trend as upward move
        diff = max - min

        l1 = max - 0.236 * diff
        l2 = max - 0.386 * diff
        l3 = max - 0.618 * diff
        # Extension levels
        l4 = max + 0.236 * diff
        l5 = max + 0.386 * diff
        l6 = max + 0.618 * diff
        l7 = max + 1 * diff
        l8 = max + 1.38 * diff

        print ("\n\nCurrent price is: ",price)
        print ("\n\nFibonacci levels considering an upward trend.")
        print ("\n\nLevel \tRetracement Prices \t\tLevel \tExtension Prices")
        print ("0% \t", max," \t\t\t23.6% \t",l4)
        print ("23.6% \t", l1," \t\t38.6% \t",l5)
        print ("38.2% \t", l2," \t\t61.8% \t",l6)
        print ("61.8% \t", l3," \t\t100% \t",l7)
        print ("100%  \t", min,"\t\t\t138.2% \t",l8)
    else:
        # Making Fibonacci Levels considering original trend as downward move
        diff = max - min

        l1 = min + 0.236 * diff
        l2 = min + 0.386 * diff
        l3 = min + 0.618 * diff
        # Extension levels
        l4 = min - 0.236 * diff
        l5 = min - 0.386 * diff
        l6 = min - 0.618 * diff
        l7 = min - 1 * diff
        l8 = min - 1.38 * diff

        print ("\n\nCurrent price is: ",price)
        print ("\n\nFibonacci levels considering an upward trend.")
        print ("\n\nLevel \tRetracement Prices \t\tLevel \tExtension Prices")
        print ("0% \t", max," \t\t\t23.6% \t",l4)
        print ("23.6% \t", l1," \t\t38.6% \t",l5)
        print ("38.2% \t", l2," \t\t61.8% \t",l6)
        print ("61.8% \t", l3," \t\t100% \t",l7)
        print ("100%  \t", min,"\t\t\t138.2% \t",l8)
    return None

def min_max_processing(data):
    a = int(input("Enter the number of days to be analysed: "))
    df = data.tail(a)

    # Processing data
    #------------------------------------------------------------------
    
    # Converting dataframe to numpy array.
    close = df[["Close"]].to_numpy()

    # These lines make a list of all indices where the value of close[i] is greater than both of its neighbours.
    peaks = np.where((close[1:-1] > close[0:-2]) * (close[1:-1] > close[2:]))[0] + 1
    dips = np.where((close[1:-1] < close[0:-2]) * (close[1:-1] < close[2:]))[0] + 1

    # Endpoints are not checked, which only have one neighbour each.
    # The extra +1 at the end is necessary because where finds the indices within the slice y[1:-1], not the full array y.
    # The [0] is necessary because where returns a tuple of arrays, where the first element is the array we want.

    # Plotting Data
    #------------------------------------------------------------------
    x = np.linspace(0,10,a)
    plt.plot (x,close)
    plt.plot (x[peaks], close[peaks], 'x', label = "Local Peak")
    plt.plot (x[dips], close[dips], 'o', label = "Local Dip")
    plt.title("Minima and Maxima plot of the Stock")
    plt.legend()
    plt.show()
    return None

def moving_averages_processing(data):
    a = int(input("Enter the number of days to be analysed: "))
    data = data.tail(a)
    data.index = pd.to_datetime(data.index)

    x = int(input("How many days to consider while calculating the short term averages? - "))
    y = int(input("How many days to consider while calculating the long term averages? - "))
    #-------------------------------------
    # Processing the data
    #-------------------------------------

    # Short term
    weights = np.arange(1, (x+1))
    wma = data['Close'].rolling(x).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    sma = data['Close'].rolling(x).mean()
    ema = data['Close'].ewm(span=x, adjust = False).mean()

    # Long term
    weights = np.arange(1, (y+1))
    wma_l = data['Close'].rolling(y).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    sma_l = data['Close'].rolling(y).mean()
    ema_l = data['Close'].ewm(span=y, adjust=False).mean()

    #----------------------------------------
    # Displaying results
    #----------------------------------------
    choice = int(input('''
    Choose the type of Averaging for analysis.
    1)WMA (Weighted)
    2)SMA (Simple)
    3)EMA (Expoenential)
    '''))

    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label="Price")

    if choice == 1:
        plt.plot(wma, label="{}-Day WMA".format(x))
        plt.plot(wma_l, label="{}-Day WMA".format(y))
    elif choice == 2:
        plt.plot(sma, label="{}-Day SMA".format(x))
        plt.plot(sma_l, label="{}-Day SMA".format(y))
    elif choice == 3:
        plt.plot(ema, label="{}-Day EMA".format(x))
        plt.plot(ema_l, label="{}-Day EMA".format(y))
    #Labelling
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    return None

def prophet_processing(data):
    warnings.filterwarnings("ignore")
    a = int(input("Enter the number of days to be analysed: "))
    df = data.tail(a)
    #-----------------------------------------------------
    # Simple plotting of Stock Price
    #-----------------------------------------------------
    # First Subplot
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(df["Close"])
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Stock Price")
    ax1.set_title("Close Price History")
    
    # Second Subplot
    ax1.plot(df["High"], color="green")
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Stock Price")
    ax1.set_title("High Price History")
    
    # Third Subplot
    ax1.plot(df["Low"], color="red")
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Stock Price")
    ax1.set_title("Low Price History")

    # Fourth Subplot
    ax2.plot(df["Volume"], color="orange")
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Stock Price")
    ax2.set_title("Volume History")
    plt.show()
    
    # Using Prophet now
    #----------------------------------------------------

    # Droping the extra columns
    ph_df = df.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1)
    ph_df.rename(columns={'Date':'ds','Close':'y'}, inplace=True)
    print(ph_df.head())

    # Creating the Model and fitting the data
    m = Prophet()
    m.fit(ph_df)
    # Creating Future dates
    future_prices = m.make_future_dataframe(periods=365)
    # Predicting Prices
    forecast = m.predict(future_prices)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    #--------------------------------------------
    # Plotting the overall forecast data
    #--------------------------------------------

    fig = m.plot(forecast)
    ax1 = fig.add_subplot(111)
    ax1.set_title("Stock Price Forecast", fontsize=16)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Close Price", fontsize=12)
    plt.show()

    # Showing various components of the plot
    fig2 = m.plot_components(forecast)
    plt.show()

    #-----------------------------------------------
    # Plotting the monthly data prediction
    #-----------------------------------------------
    m = Prophet(changepoint_prior_scale=0.01)
    m.fit(ph_df)
    future = m.make_future_dataframe(periods=12, freq='M')
    fcst = m.predict(future)
    fig = m.plot(fcst)
    plt.title("Monthly Prediction \n 1 year time frame")
    plt.show()
    
    # Showing various components of the plot
    fig = m.plot_components(fcst)
    plt.show()
    return None

def lstm_processing(data):
    c = int(input ("Enter the number of days to be analysed: "))
    a = int(input("Number of days for short term memory: "))
    
    df = data.tail(c)
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

    quote = data
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
    return None
