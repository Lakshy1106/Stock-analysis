import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------
# Taking Inputs
#------------------------------------------------------------------

a = int(input("How many days must be included in dataset: "))
b = input("What is the stock quote: ")
c = input("What is file extension: ")
if c == 'csv':
    df = pd.read_csv("Database/{}.csv".format(b), index_col ="Date")
else:
    df = pd.read_excel("Database/{}.xlsx".format(b), index_col ="Date")
df = df.tail(a)

#------------------------------------------------------------------
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

#------------------------------------------------------------------
# Plotting Data
#------------------------------------------------------------------
x = np.linspace(0,10,a)
plt.plot (x,close)
plt.plot (x[peaks], close[peaks], 'x', label = "Local Peak")
plt.plot (x[dips], close[dips], 'o', label = "Local Dip")
plt.title("Minima and Maxima plot of the Stock")
plt.legend()
plt.show()
