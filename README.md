# Prediction of Stock Prices Using Time-Frequency Analysis and Convolutional Neural Networks

## Introduction
This is a short research idea I explored for a couple of weeks for a novel way of approaching the allure of stock prediction that attracts every machine learning novice. The idea this time is to perform a time-frequency analysis on the stock price data using an STFT (Short Time Fourier Transform) and then passing this processed data as an image to a Convolutional Neural Network to try and predict whether the model can predict whether the stock price will go up or down based on the processed data that it has been fed. The underlying assumption here is that using STFT's might reveal some underlying features about the changing stock prices that a Deep Learning model might not be able to capture on its own through normal training. Other wierd choices such as the data chosen for training are purely a result of financial and physical constraints. This short research endeavor hopes to uncover the practically of such novel methods (it's not really that good) and suggest some future prospects which I might personally attempt but really anybody is welcome to do so. 

## Datasets

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
### S&P-Data

The S&P-Data Data set consists of 1-minute interval closing price of the S&P 500 index from 2008-2024 that I extracted from a bloomberg terminal by hand, copying and pasting on to an excel sheet while listening to a podcast. Because of the vastness of the dataset, I randomly chose 50,000 datapoints from the dataset to train the model.

Each datapoint in the S&P-Data dataset comprises of the closing prices recorded over the predecing 381*5 minutes, i.e the last 5 working days. The Y-Value is the percentage change in the stock price after 381 minutes, i.e. one working day.

![image](https://github.com/user-attachments/assets/1f44ffbc-9912-4ef0-ba20-ea44210c5a83)<br>
Visual representation of S&P-Data datapoint
<br>

I really didn't need to do alot of processing for this dataset so not alot of code was needed to process this data.

## Data Processing

The X values from both datasets was converted to spectographs using the Short Time Fourier Transform (STFT) function provided by the scipy library. This was done as the STFT is effective in providing time localized frequency information for signals whose frequency components may vary over time. 
<br>
Originally the spectogram resolution was supposed to be 640x480 pixels. However, because of limited compute resources (kaggle's free tier) this resolution was halved to 320x240 pixels. The following code illustrates how made the spectograms:

```
def data_to_Stft(Name):
    detrended = signal.detrend(X_Val[Name])
    f, t, Sxx = signal.stft(detrended, fs=1, nperseg=512, nfft=1024)
    plt.pcolormesh(t, f, abs(Sxx), shading='gouraud')
    plt.ylim(0,0.01)
    plt.axis('off')
    plt.subplots_adjust(bottom=0)
    plt.subplots_adjust(top=1)
    plt.subplots_adjust(right=1)
    plt.subplots_adjust(left=0)
    plt.savefig(str(Name + 1) + '.png', bbox_inches='tight', pad_inches=0)
    if Name % 100 == 0:
        print(f"{Name}/{len(X_Val)}")```

```
<br>
![image(1)](https://github.com/user-attachments/assets/9b936988-21ef-41ee-b241-159ccd07945f)<br>
an example of a spectogram produced for training

## Model





