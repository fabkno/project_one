import numpy as np
import pandas as pd
print pd.__version__
import sys
import datetime as dt
from singleStockScan import ScanModel, ModelPrediction
#ob = ScanModel(ModelType='SVM',n_jobs=-2)
#ob.StockGridModelingAll()
#print "all required stocks updated"

ob = ModelPrediction('DowJones.csv')
#ob.ComputeStockModelsAll()
ob.PredictStocksAll()


#sys.exit()
#from updater import StockUpdater
#ob = StockUpdater()

#ob.update_stock_prizes()
#ob.update_all()

