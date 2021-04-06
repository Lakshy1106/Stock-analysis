# Importing libraries
#---------------------------------------------

import pandas as pd
from tools import fibonacci_processing, min_max_processing, lstm_processing , moving_averages_processing, prophet_processing

# Starting main program
#---------------------------------------------

def choice_taker():
    choices = ['fibonacci_retracement','lstm','minima_maxima','moving_averages','prophet']
    
    print('''
--------------------------------------------------------------------------------------------
PROGRAM START
--------------------------------------------------------------------------------------------
This application has the following tools to offer please choose one that suits best to your needs.
        
1. Fibonacci retracement
2. LSTM analysis (currently works only on windows systems or linux systems with nvidia gpu)
3. Minima Maxima analysis
4. Moving averages analysis.
5. Prophet analysis. (An overall analysis of your data and predictions from prebuilt fbprophet tool)''')
    
    choice = int(input("Please input the number of your selection here : "))
    
    return choices[choice-1]

def data_finder(choice):
    if choice == 'fibonacci_retracement':
        # Taking input filename and making address
        file_name = input("Please enter the name of the file of stock data you want to analyse alongwith extension : ")
        address = "../Database/{}".format(file_name)
        
        # Converting file name in list to detect file type
        file_name_list = file_name.split('.')
        
        #Extracting data based on file type.
        if 'xlsx' in file_name_list:
            data = pd.read_excel(address, index_col = "Date")
        elif 'csv' in file_name_list:
            data = pd.read_csv(address, index_col = "Date")
    
    elif choice == 'lstm':
        # Taking input filename and making address
        file_name = input("Please enter the name of the file of stock data you want to analyse alongwith extension : ")
        address = "../Database/{}".format(file_name)
        
        # Converting file name in list to detect file type
        file_name_list = file_name.split('.')
        
        #Extracting data based on file type.
        if 'xlsx' in file_name_list:
            data = pd.read_excel(address, index_col = "Date")
        elif 'csv' in file_name_list:
            data = pd.read_csv(address, index_col = "Date")
    
    elif choice == 'minima_maxima':
        # Taking input filename and making address
        file_name = input("Please enter the name of the file of stock data you want to analyse alongwith extension : ")
        address = "../Database/{}".format(file_name)
        
        # Converting file name in list to detect file type
        file_name_list = file_name.split('.')
        
        #Extracting data based on file type.
        if 'xlsx' in file_name_list:
            data = pd.read_excel(address, index_col = "Date")
        elif 'csv' in file_name_list:
            data = pd.read_csv(address, index_col = "Date")
    
    elif choice == 'moving_averages':
        # Taking input filename and making address
        file_name = input("Please enter the name of the file of stock data you want to analyse alongwith extension : ")
        address = "../Database/{}".format(file_name)
        
        # Converting file name in list to detect file type
        file_name_list = file_name.split('.')
        
        #Extracting data based on file type.
        if 'xlsx' in file_name_list:
            data = pd.read_excel(address, index_col = "Date")
        elif 'csv' in file_name_list:
            data = pd.read_csv(address, index_col = "Date")
    
    elif choice == 'prophet':
       # Taking input filename and making address
        file_name = input("Please enter the name of the file of stock data you want to analyse alongwith extension : ")
        address = "../Database/{}".format(file_name)
        
        # Converting file name in list to detect file type
        file_name_list = file_name.split('.')
        
        #Extracting data based on file type.
        if 'xlsx' in file_name_list:
            data = pd.read_excel(address)
        elif 'csv' in file_name_list:
            data = pd.read_csv(address)
    return data

def main():
    choice = choice_taker()
    data_frame = data_finder(choice)
    if choice == 'fibonacci_retracement':
        fibonacci_processing(data_frame)
    elif choice == 'lstm':
        lstm_processing(data_frame)
    elif choice == 'minima_maxima':
        min_max_processing(data_frame)
    elif choice == 'moving_averages':
        moving_averages_processing(data_frame)
    elif choice == 'prophet':
        prophet_processing(data_frame)
    return None

# Calling main to start program
#--------------------------------------------
main()
