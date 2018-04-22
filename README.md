# project_one
stock prediction

required packages besides obvious ones:

- pandas-datareader https://pandas-datareader.readthedocs.io/en/latest/

- pandas 0.22

- flask-restful 

- sqlalchemy 1.2.3


- flask-jsonpify 1.5

- scheduler 0.5

- tensorflow 1.0


- pip install flask-jsonpify 



- keras v. ...
Some fixes:

- incorporated national holidays. So far Germany and US



Some notes:

- create list of stocks for which update of stock prize didn't work and recheck automatically DONE

- get statistics of missing number in stocks

- check if it's worth adding relative min/max different windows of macdh 

- check what happens when stocks are updated before stock exchange has closed

- evtl. incoporate slope MACD, ADX, SMA, ....


version 0.1

- Stockupater for DAX, MDAX, SDAX, TecDAX, EUROSTOXX 50, DowJones and NASDAQ 100

- Chartmarkers: MACD, PDM, true range, ATR, williams oscillators, moving averages 200,100,50 ... , ADX, CCI, bollinger bands, RSI, PVO, TRIX, RSV

- Classification for n businessdays (3, 5 and 10)

- Model prediction through SVM or RFC

- daily updates (still some bugs), model training and prediction 

- predictions saved as pandas database and sql accessible database

- logfile class 

- restAPI to access sql database


to do for version 0.2

- create official python package with setup.py and path setting (init.py)

- try different scheduler class or write own class

- add csv file to individually select chart marker for stocks (stored in csv file)

- hyperparameter random search

- add major european stocks and S&P500

- add Deep Network (3-5 layers) and LSTM with 1-2 layers 

- add in restAPI other prediction windows (at v0.1 only 3 days)

- add more stock markers 

- add currencies, stock indices, resources

- check for n day correlations between stocks

