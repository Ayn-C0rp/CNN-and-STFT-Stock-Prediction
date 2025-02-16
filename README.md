# Prediction of Stock Prices Using Time-Frequency Analysis and Convolutional Neural Networks

## Introduction
This is a short research idea I explored for a couple of weeks for a novel way of approaching the allure of stock prediction that attracts every machine learning novice. The idea this time is to perform a time-frequency analysis on the stock price data using an STFT (Short Time Fourier Transform) and then passing this processed data as an image to a Convolutional Neural Network to try and predict whether the model can predict whether the stock price will go up or down based on the processed data that it has been fed. The underlying assumption here is that using STFT's might reveal some underlying features about the changing stock prices that a Deep Learning model might not be able to capture on its own through normal training. Other wierd choices such as the data chosen for training are purely a result of financial and physical constraints. This short research endeavor hopes to uncover the practically of such novel methods (it's not really that good) and suggest some future prospects which I might personally attempt but really anybody is welcome to do so. 

## Data and Processing

I used two different datasets to train the model and compared the performance of both of them. They are:<br>

### Rand-Data:

The Rand-Data dataset comprises 2-minute interval data spanning across a period of 60 days from all the individual stocks in the S&P 500. This was done because I felt that taking data from longer time frames would mean that other unknown factors that could not be captured just from the stock data itself would be effecting the stock price. I also had to clean and filter the data as the free YFinance data had alot of gaps and inconsistencies (surprise surpise). By the end of filtering, I had approximately 20,000 datapoints to train the model on. 

Each datapoint in the Rand-Data dataset includes 2-minute interval closing price of a random (unknown stock) for the last 15 days (the X value of the data set) and the corresponding percentage in stock price after 5 working days (the Y value of the data set). Such data for the X value chosen as this gave a good balance between the temporal resolution available for each individual spectogram and the amount of spectograms that could be extracted from each of the stocks I had access to.
<br>
<br>
![alt text](https://github.com/Ayn-C0rp/CNN-and-STFT-Stock-Prediction/blob/main/Screenshot%202025-02-16%20131852.png)<br>
(Visual Representation of a datapoint from Rand-Data)
<br>
<br>
<br>

Here is the code that I used to extract the data for the rand-data dataset:<br>
<br>
Importing and setting up libraries:
```
!pip install yfinance
import datetime
import os as sys
from datetime import date
from IPython.display import clear_output
import yfinance as stock
import pandas as pd
data = pd.read_csv('/kaggle/input/s-and-p500/S_P 500 Companies (Standard and Poor 500) - basics.csv')
Symbols = data['Symbol'].tolist()
print(len(Symbols))
# Import the 'multiprocessing' module to work with multi-processing features.
import multiprocessing

# Use 'multiprocessing.cpu_count()' to determine the number of available CPU cores.
cpu_count = multiprocessing.cpu_count()

# Print the number of CPU cores available on the system.
print(cpu_count)

```
<br>
Setting up functions to get the data:

```
def TwentyOneDaysBehind(Ctime):
    CountDays = 0
    if Ctime.weekday() == 5 or Ctime.weekday() == 6:
        return None, None
    i = 1
    while(CountDays < 15):
        DateToCheck = Ctime - datetime.timedelta(i)
        if(DateToCheck.weekday() == 5 or DateToCheck.weekday() == 6):
            i+= 1
        else:
            i+=1
            CountDays += 1
    return DateToCheck, Ctime


def SevenDaysAhead(Ctime):
    CountDays = 0
    if Ctime.weekday() == 5 or Ctime.weekday() == 6:
        return None, None
    i = 1
    while(CountDays < 5):
        DateToCheck = Ctime + datetime.timedelta(i)
        if(DateToCheck.weekday() == 5 or DateToCheck.weekday() == 6):
            i+= 1
        else:
            i+=1
            CountDays += 1
    return DateToCheck, Ctime


def OneDaysAhead(Ctime):
    CountDays = 0
    if Ctime.weekday() == 5 or Ctime.weekday() == 6:
        return None, None
    i = 1
    while(CountDays < 2):
        DateToCheck = Ctime + datetime.timedelta(i)
        if(DateToCheck.weekday() == 5 or DateToCheck.weekday() == 6):
            i+= 1
        else:
            i+=1
            CountDays += 1
    return DateToCheck, Ctime


def CheckIfStockIncreasing(Ctime, tick):
    Ahead1, Ctime = OneDaysAhead(Ctime)
    List = list(tick.history(start=Ctime, end=Ahead1, period='1d')['Close'])
    l = []
    StockState = -2
    if len(List) == 2:
        l = List
        if List[1] > List[0] * 1.05:
            StockState = 1
        elif List[1] < List[0] * 0.95:
            StockState = -1
        else:
            StockState = 0

    return StockState, l
    
def GetXVal(Ctime, tick):
    Behind21, Ctime = TwentyOneDaysBehind(Ctime)
  
    
    return list((tick.history(start=Behind21, end=Ctime, interval='2m'))['Close'])

def CollectAllPossibleStockData(symbol):
    Ctime = date.today()
    Delta2 = datetime.timedelta(1)
    Time_1 = Ctime
    tick = stock.Ticker(str(symbol))
    for i in range(60):
        YVal = -2
        YVal2 = []
        XVal = []
        YVal, YVal2 = CheckIfStockIncreasing(Time_1, tick)
        XVal = GetXVal(Time_1, tick)
       
        if(YVal != -2 and XVal):    
            Data_File.write(str(XVal)) 
            Data_File.write('\t' + str(YVal)+ '\t' + str(YVal2) + '\n')
        Time_1 =   Time_1 - Delta2
    if(int(Symbols.index(symbol))%(len(Symbols)//100) == 0):
           print(str(Symbols.index(symbol)) + '/' + str(len(Symbols)))
      
```
<br>

And Finally the code to output the final data file:

```
Data_File = open('Data.txt', 'w')
from concurrent.futures import ProcessPoolExecutor
 

if __name__ == '__main__':
    # report a message
    print('Starting task...')
    # create the process pool
    with ProcessPoolExecutor(64) as exe:
        # perform calculations
        results = exe.map(CollectAllPossibleStockData, Symbols)
    # report a message
    print('Done.')```

```

