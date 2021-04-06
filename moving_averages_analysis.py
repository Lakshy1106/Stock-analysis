#-----------------------------------
# Importing modules
#----------------------------------

from operator import add
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#-------------------------------------
# Gathering data
#-------------------------------------
b = input("Enter the stock quote - ")
c = input("What is the file format? (csv or xlsx): ")
a = int(input("Enter the number of days to be analysed: "))
address = "Database/{}.{}".format(b,c)

if c == 'csv':
	data = pd.read_csv(address, index_col="Date")
else:
	data = pd.read_excel(address, index_col="Date")
data.tail(a)
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
