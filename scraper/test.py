import pandas as pd
import sys
import difflib

test = pd.ExcelFile('/localscratch/data/project_one/scraper/yahoo_ticker_sept17.xlsx')
companies = pd.read_pickle('/localscratch/data/project_one/scraper/companies_by_branches.p')

data = test.parse('Stock')

data_new =  data[['Yahoo Stock Tickers','Unnamed: 1','Unnamed: 4']]
data_new = data_new.rename(str,columns={"Yahoo Stock Tickers": "Ticker", 'Unnamed: 1': "Name",'Unnamed: 4':'Country'})


data_new = data_new.drop_duplicates('Ticker')

data_new= data_new.dropna(axis=0)

data_new = data_new[1:]
data_new.reset_index(drop=True,inplace=True)