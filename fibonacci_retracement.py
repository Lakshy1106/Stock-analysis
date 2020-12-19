import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#------------------------------------------------------------------
# Taking Inputs
#------------------------------------------------------------------

b = input("Enter the stock quote: ")
c = input("What is the file format? (csv or xlsx): ")
a = int(input ("\nEnter the number of days to be analysed: "))
address = "Database/{}.{}".format(b,c)

if c == 'csv':
	df = pd.read_csv(address, index_col="Date")
else:
	df = pd.read_excel(address, index_col="Date")

# Reading last 'a' rows only.
df = df.tail(a)
df.index = pd.to_datetime(df.index)

#------------------------------------------------------------------
# Processing data
#------------------------------------------------------------------

min = df["Close"].min()
max = df["Close"].max()

price = df["Close"].last('1D')

# Plotting Close prices graph for upward trend
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
	print ("23.6% \t", l1," \t\t\t38.6% \t",l5)
	print ("38.2% \t", l2," \t\t61.8% \t",l6)
	print ("61.8% \t", l3," \t\t100% \t",l7)
	print ("100%  \t", min," \t\t\t138.2% \t",l8)
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
	print ("23.6% \t", l1," \t\t\t38.6% \t",l5)
	print ("38.2% \t", l2," \t\t61.8% \t",l6)
	print ("61.8% \t", l3," \t\t100% \t",l7)
	print ("100%  \t", min," \t\t\t138.2% \t",l8)
