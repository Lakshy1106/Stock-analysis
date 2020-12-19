#-----------------------------------------------------
# Importing Libraries
#-----------------------------------------------------
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet

warnings.filterwarnings("ignore")
# Plotting style to be used.
plt.style.use("seaborn-whitegrid")

#-----------------------------------------------------
# Collecting data
#-----------------------------------------------------
a = int(input("What is your time frame of analysis?: "))
b = input("Enter the stock quote: ")
c = input("Enter the file format: ")
address = "Database/{}.{}".format(b,c)
if c == 'csv':
	df = pd.read_csv(address, index_col="Date")
else:
	df = pd.read_excel(address, index_col="Date")

df = df.tail(a)

#-----------------------------------------------------
#  Simple plotting of Stock Price
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

#---------------------------------------------------
# Using Prophet now
#----------------------------------------------------

# Droping the extra columns
ph_df = df.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1)
ph_df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
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

#-----------------------------------------------
# End of program
#-----------------------------------------------
