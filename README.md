# Prediction of Stock Prices Using Time-Frequency Analysis and Convolutional Neural Networks

## Introduction
This is a short research idea I explored for a couple of weeks for a novel way of approaching the allure of stock prediction that attracts every machine learning novice. The idea this time is to perform a time-frequency analysis on the stock price data using an STFT (Short Time Fourier Transform) and then passing this processed data as an image to a Convolutional Neural Network to try and predict whether the model can predict whether the stock price will go up or down based on the processed data that it has been fed. The underlying assumption here is that using STFT's might reveal some underlying features about the changing stock prices that a Deep Learning model might not be able to capture on its own through normal training. Other wierd choices such as the data chosen for training are purely a result of financial and physical constraints. This short research endeavor hopes to uncover the practically of such novel methods (it's not really that good) and suggest some future prospects which I might personally attempt but really anybody is welcome to do so. 

## Data and Processing

I used two different datasets to train the model and compared the performance of both of them. They are:<br>

### Rand-Data:

The Rand-Data dataset comprises 2-minute interval data spanning across a period of 60 days from a selection of 6928 stocks (all the stocks available on the YFinance API). This was done because I felt that taking data from longer time frames would mean that other unknown factors that could not be captured just from the stock data itself would be effecting the stock price. The large number of stocks used is because I couldn't find (afford) enough data from just a few stocks. I also had to clean and filter the data as the free YFinance data had alot of gaps and inconsistencies (surprise surpise). By the end of filtering, I had approximately 20,000 datapoints to train the model on. I'll get into the specifics of each datapoint in a bit.

