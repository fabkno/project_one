import numpy as np
import time
import datetime
from updater import StockUpdater
from shorts import ShortsUpdater

from singleStockScan import ModelPrediction
from merge_to_sql_database import MergeToSQL

import schedule

#def update_shorts():
	#ob = ShortsUpdater()
	#ob.update_from_web()

def update_stock_prizes(Countries):

 	print "Countries:" , Countries

 	if Countries == "Europe":
		#print "do nothing"
 		StockUpdater(duration=3,FileNameListOfCompanies='EUstocks.csv').update_all()
 		StockUpdater(duration=5,FileNameListOfCompanies='EUstocks.csv').update_stock_classification()

			
 	elif Countries == "US":

 		StockUpdater(duration=3,FileNameListOfCompanies='USstocks.csv').update_all()
 		StockUpdater(duration=5,FileNameListOfCompanies='USstocks.csv').update_stock_classification()


 	return

def update_daily_prediction(Countries):

	print "Start updating daily predictions for ",Countries

	if Countries == 'Europe':
		print "europe"
		ob2 = ModelPrediction(duration=3,FileNameListOfCompanies='EUstocks.csv')
		ob2.UpdateStockModelAndPredict(double_save=True)
		sql = MergeToSQL(duration=3)
		sql.compare_databases(duration=3,ListOfLabels = ob2.ListOfCompanies['Yahoo Ticker'].tolist(),ListOfPredictionDays =[datetime.datetime.today().date()])
		#MergeToSql(duration=5).compare_databases(ListOfLabels = ob2.ListOfCompanies['Yahoo Ticker'].tolist(),ListOfPredictionDays =[datetime.datetime.today().date()])
		print "database for EU stocks updated"


	elif Countries == 'US':

		ob2 = ModelPrediction(duration=3,FileNameListOfCompanies='USstocks.csv')
		ob2.UpdateStockModelAndPredict(double_save=True)

		sql = MergeToSQL(duration=3)
		sql.compare_databases(duration=3,ListOfLabels = ob2.ListOfCompanies['Yahoo Ticker'].tolist(),ListOfPredictionDays =[datetime.datetime.today().date()])
		print "database for US stocks updated"		

schedule.every().monday.at("18:00").do(update_stock_prizes,'Europe')
schedule.every().monday.at("19:00").do(update_daily_prediction,'Europe')
schedule.every().monday.at("22:30").do(update_stock_prizes,'US')
schedule.every().monday.at("23:30").do(update_daily_prediction,'US')

schedule.every().tuesday.at("18:00").do(update_stock_prizes,'Europe')
schedule.every().tuesday.at("19:00").do(update_daily_prediction,'Europe')
schedule.every().tuesday.at("22:30").do(update_stock_prizes,'US')
schedule.every().tuesday.at("23:30").do(update_daily_prediction,'US')

schedule.every().wednesday.at("18:00").do(update_stock_prizes,'Europe')
schedule.every().wednesday.at("19:00").do(update_daily_prediction,'Europe')
schedule.every().wednesday.at("22:30").do(update_stock_prizes,'US')
schedule.every().wednesday.at("23:30").do(update_daily_prediction,'US')

schedule.every().thursday.at("18:00").do(update_stock_prizes,'Europe')
schedule.every().thursday.at("19:00").do(update_daily_prediction,'Europe')
schedule.every().thursday.at("22:30").do(update_stock_prizes,'US')
schedule.every().thursday.at("23:30").do(update_daily_prediction,'US')

schedule.every().friday.at("18:00").do(update_stock_prizes,'Europe')
schedule.every().friday.at("19:00").do(update_daily_prediction,'Europe')
schedule.every().friday.at("22:30").do(update_stock_prizes,'US')
schedule.every().friday.at("23:30").do(update_daily_prediction,'US')


#update_daily_prediction('Europe')
while True:
	schedule.run_pending()
	time.sleep(600)
