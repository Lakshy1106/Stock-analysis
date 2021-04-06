# Stock-market-analysis

- This project is currently being developed will be updated with more tools for analysis and present codes will be refined.
- Latest update details.
	- fixes applied in lstm_analysis.py and prophet_analysis.py

This repository contains the  codes I have created for stock market analysis, it is meant for public use and database which will be analysed must be downloaded by the user.
(A good website for downloading databases is yahoo finance.)

Preparing to use these codes.
1) Download all the codes in your computer.
2) Extract the zip file.
3) To add your database download the data files in csv or excel format from any website you choose. (yahoo finance has been tested and must be preferred)
4) Use the Database folder inside the folder you extracted to save all your csv/xlsx files. (It already has a test file for you.)
5) When running the code put the correct file name of the data file and code must analyse the data.

How to run these codes.
1) To run these codes you will need python installed in your system.
2) For installing python on windows please download the installer file from their official website. https://www.python.org/downloads/windows/
Note - for linux users python must be ususlally installed by default in your computer.
3) For running these codes you must have the following python packages pre installed in your system.
	1) Tensorflow
	2) Keras
	3) Numpy
	4) Matplotlib
	5) Pandas

4) For installing above packages use the followng code in your command prompt - pip install *insert package name here*

5) After dependecies are installed run the python files.
	1) In windows an IDLE must be installed with the python package. Open the code file using the IDLE and run the code.
	2) In Linux open terminal and use this command - python3 *insert code filename here*

Extra notes
1) Download the max size of data for the stock you need to analyse.(You can choose the range of data to be analysed from the pprogram itself.)
2) Currently only csv and excel are supported.
3) Only the Close price of the particular stock is currently analysed, I will be adding a feature for analysing different parameters like Open price, High Low extra soon.
