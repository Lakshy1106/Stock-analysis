from operator import add
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time


b = input("Enter the stock name - ")
address = "Database/1 Month/{}.csv".format(b)
df = pd.read_csv(address, index_col="Date")
df.index = pd.to_datetime(df.index)

fig, ax = plt.subplots()
ax.plot(df['Close'], color='black')

min = df["Close"].min()
max = df["Close"].max()

# Fibonacci Levels considering original trend as upward move
diff = max - min
level1 = max - 0.236 * diff
level2 = max - 0.382 * diff
level3 = max - 0.618 * diff

print ("Level        Price")
print ("0          ", max)
print ("0.236      ", level1)
print ("0.382      ", level2)
print ("0.618      ", level3)
print ("1          ", min)

ax.axhspan(level1, min, alpha=0.4, color='lightsalmon')
ax.axhspan(level2, level1, alpha=0.5, color='palegoldenrod')
ax.axhspan(level3, level2, alpha=0.5, color='palegreen')
ax.axhspan(max, level3, alpha=0.5, color='powderblue')

plt.ylabel("Price")
plt.xlabel("Date")
plt.show()