from pandas.tseries.offsets import CustomBusinessDay
from pandas import read_csv

d = read_csv('../data/holidays/holidays.csv')


holidays_ger = d.loc[d['Country'] == 'Germany','Date'].values
holidays_us = d.loc[d['Country'] == 'US','Date'].values

bday_ger = CustomBusinessDay(holidays=holidays_ger)
bday_us = CustomBusinessDay(holidays=holidays_us)
