import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import os,shutil,sys
from pandas_datareader._utils import RemoteDataError
import stockstats

	# 		#find index with NAN
	# 		indsInput = InputData.loc[InputData.notnull().all(axis=1)].index.tolist()
	# 		indsOutput = OutputData.loc[OutputData.notnull().all(axis=1)].index.tolist()
			
	# 		#find intersection betwen both index lists
	# 		inds_final = list(set(indsInput) & set(indsOutput))

class StockUpdater(object):
	"""

	Add description

	"""
	def __init__(self,FileNameListOfCompanies=None,PathData=None):


		'''
		Parameters
		----------------
		PathData : string (default None) give manually working directory 

		FileNameListOfCompanies : string (default None) if not full list of companies is considered to update give filename.csv of indented list
						  the list has to be stored in "../data/company_lists/"


		ListOfChartFeatures: "GDXX" = moving average and number of days
							 "BB_20_2" = bolling bands (tau = 20 and k=2)
							 "RSI_14" = relative strength index (last 14 days)
							 "ADX" = average directional index
							 "MACD" = moving average convergence/divergence ###check number of days
							 "MAXxx" = moving maximum of xx days
							 "MINxx" = moving minimum of xx days
		'''
		self.ListOfChartFeatures = ['GD200','GD100','GD50','GD38','BB_20_2','RSI_14','ADX','MACD','MAX20','MAX65','MAX130','MAX260','MIN20','MIN65','MIN130','MIN260']

		'''
		PrizeThresholds : threshold to ategorize relative (in percent) stock evolution within N days

		'''
		self.PrizeThresholds=[-5,-2.5,0,2.5,5]
		self.duration = 10 #duration for which to compute classification 

		if PathData is None:
			self.PathData = os.path.dirname(os.getcwd())+'/data/'
		else:
			self.PathData = PathData
		
		#check for input ListOfCompanies
		if FileNameListOfCompanies is None:
			self.FileNameListOfCompanies = 'full_list.csv'

		else:
			if os.path.isfile(self.PathData+'company_lists/'+FileNameListOfCompanies) is True:
				self.FileNameListOfCompanies = FileNameListOfCompanies
			else:
				raise ValueError('List: '+FileNameListOfCompanies + ' does not exists in' +self.PathData + 'company_lists/')

		
		self.ListOfCompanies = pd.read_csv(self.PathData+'company_lists/'+self.FileNameListOfCompanies,index_col='Unnamed: 0')
		
		#check if directories exists otherwise create them
		if os.path.exists(self.PathData+'raw/stocks/') is False:
			os.makedirs(self.PathData+'raw/stocks/')

		if os.path.exists(self.PathData+'raw/stocks/backup/') is False:
			os.makedirs(self.PathData+'raw/stocks/backup/')

		if os.path.exists(self.PathData+'chart/stocks/') is False:
			os.makedirs(self.PathData+'chart/stocks/')

		if os.path.exists(self.PathData+'classification/stocks/') is False:
			os.makedirs(self.PathData+'classification/stocks/')


	def update_all(self):


		self.update_stock_prizes()
		self.update_chart_markers()
		self.update_stock_classification()

	def update_stock_prizes(self):

		"""
		Update stock prizes given in ListOfCompanies using yahoo finance 

		If there is data to update the old file is backuped to .../backup/stockTicker.p so the backup is good for one business day

		"""
		print "Start updating stock prizes"
		print "--------------------------------------\n"

		self.UpdateTimeEnd = datetime.datetime.today().date()
		print "Today is ",self.UpdateTimeEnd,"\n"

		for stocklabel in self.ListOfCompanies['Yahoo Ticker']:

			if os.path.isfile(self.PathData + 'raw/stocks/'+stocklabel+'.p'):
				StockValue = pd.read_pickle(self.PathData + 'raw/stocks/'+stocklabel+'.p')
				
				self.UpdateTimeStart = StockValue.tail(1)['Date'].tolist()[0].date()				

				#if stock has been updated at the same date already
				
				if self.UpdateTimeStart ==  self.UpdateTimeEnd:
					continue
				try:
					
					stock_prize = pdr.get_data_yahoo(stocklabel,self.UpdateTimeStart,self.UpdateTimeEnd)
					stock_prize.dropna(inplace=True)
					stock_prize.drop(index=stock_prize.loc[stock_prize['Volume'] == 0.0].index.tolist(),inplace=True)
					stock_prize.reset_index(inplace=True)

					#print stock_prize
					stock_prize = stock_prize.loc[stock_prize['Date']>self.UpdateTimeStart]
					
					if len(stock_prize) == 0:
						continue
					
					StockValue = pd.concat([StockValue, stock_prize], ignore_index=True)

					shutil.copy(self.PathData+'raw/stocks/'+stocklabel+'.p',self.PathData+'raw/stocks/backup/'+stocklabel+'.p')
					
					print "number of rows", len(StockValue), " for label", stocklabel
					StockValue.reset_index(inplace=True,drop=True)
					
					StockValue.to_pickle(self.PathData+'raw/stocks/'+stocklabel+'.p')
					print "Stock ",stocklabel, " updated"

				except RemoteDataError:
					print "No information for ticker ", stocklabel
					continue

			else:
				#if file is not available yet get data starting from 01/01/2000
				self.UpdateTimeStart = datetime.datetime(2000,1,1).date()
				try:
					stock_prize = pdr.get_data_yahoo(stocklabel,self.UpdateTimeStart,self.UpdateTimeEnd)
					stock_prize.drop(index=stock_prize.loc[stock_prize['Volume'] == 0.0].index.tolist(),inplace=True)
					stock_prize.dropna(inplace=True)
					stock_prize = stock_prize.reset_index()
					
					#print stock_prize
					stock_prize.to_pickle(self.PathData+'raw/stocks/'+stocklabel+'.p')

					print "Stock ",stocklabel, " updated"

				except RemoteDataError:
					print "No information for ticker ", stocklabel
					continue

		print "\nFinished updating stock prizes\n\n"


	def update_chart_markers(self):
		'''	
		update chart indicators from raw chart data
		st
		'''		
		print "Start updating chart markers"
		print "--------------------------------\n"
		for stocklabel in self.ListOfCompanies['Yahoo Ticker']:
			
			#check if raw stock data exists
			if os.path.isfile(self.PathData + 'raw/stocks/'+stocklabel+'.p'):
				rawData = pd.read_pickle(self.PathData + 'raw/stocks/'+stocklabel+'.p')	
				ChartData = self.get_chartdata(rawData)
				ChartData.dropna(inplace=True)

				ChartData.to_pickle(self.PathData+'chart/stocks/'+stocklabel+'.p')

				print "chart values for ", stocklabel, " written"
			else: 
				print "raw stock data for stock ",stocklabel, " does not exist"
				print "try running update_stock_prize() first"
		print "\nFinished updating chart markers\n\n"
	
	def update_stock_classification(self):

		'''
		update stock classification 

		'''
		print "Start updating stock classification"
		print "--------------------------------------\n"
		for stocklabel in self.ListOfCompanies['Yahoo Ticker']:
			
			#check if raw stock data exists
			if os.path.isfile(self.PathData + 'raw/stocks/'+stocklabel+'.p'):
				rawData = pd.read_pickle(self.PathData + 'raw/stocks/'+stocklabel+'.p')	
				classification = self.get_classification_output(rawData)
				classification.dropna(inplace=True)

				classification.to_pickle(self.PathData+'classification/stocks/'+stocklabel+'.p')

				print "chart values for ", stocklabel, " written"
			else: 
				print "raw stock data for stock ",stocklabel, " does not exist"
				print "try running update_stock_prize() first"

		print "\nFinished updating stock classification\n\n"


	def get_chartdata(self,rawData,ListOfChartFeatures = None):

		'''
		Parameters
		-------------
		path_raw : string, path to raw stock data

		ListOfChartFeatures : list of strings (default None)

		Returns
		-------------
		
		output : pandas DataFrame 	

		'''

		if ListOfChartFeatures is None:
			ListOfChartFeatures = self.ListOfChartFeatures
		
		#check for input
		if isinstance(rawData,basestring) == True:
			if os.path.isfile(rawData) == True:
				rawData = pd.read_csv(rawData)

				#check if rawData contains any duplicated dates
				if rawData['Date'].duplicated().any() == True:
					raise ValueError('Critical!!!! "rawData" contains duplicated Dates. Classification output is most probably inaccurate')
			else:
				raise ValueError('Input path to raw data does not exist')

		
		output = pd.DataFrame()
		output['Date'] = pd.Series(rawData['Date'],index = rawData.index)
		output['Close'] = pd.Series(rawData['Close'],index=rawData.index)
		output['Volume'] = pd.Series(rawData['Volume'],index=rawData.index)


		###change eventually by own implementation

		tmp = stockstats.StockDataFrame.retype(rawData.copy(deep=True))


		for _feature in ListOfChartFeatures:
			
			if _feature[0:2] == 'GD':	

				output[_feature] = pd.Series(self._return_relative_roll_mean(rawData,np.int(_feature[2:])),index=rawData.index)

			elif _feature[0:2] == 'BB':
				_k = np.int(_feature[-1])
				_window = np.int(_feature[3:[i for i,x in enumerate(_feature) if x=='_'][1]])

				lower,upper = self._return_relative_bollinger_bands(rawData,window_size=_window,k=_k)

				if len(lower) != len(output):
					raise ValueError('Caution length of BB bands not equal length of dates')

				output['Lower_'+_feature] = pd.Series(lower,index = rawData.index)
				output['Upper_'+_feature] = pd.Series(upper,index = rawData.index)

			elif _feature[0:3] == 'RSI':

				_window = np.int(_feature[4:])
				values = tmp.get('rsi_'+str(_window)).values		
				output[_feature] = pd.Series(values,index=rawData.index)
			elif _feature[0:3] == 'ADX':
				output[_feature] = pd.Series(tmp.get('adx').values,index=rawData.index)

			elif _feature[0:4] == 'MACD':
				output[_feature] = pd.Series(tmp.get('macd').values,index=rawData.index)

			elif _feature[0:3] == 'MAX':
				_window = np.int(_feature[3:])
			
				min_ = np.int(_window * 0.9)
				rolling_max = pd.Series.rolling(rawData['Close'],window=_window,min_periods=min_).max().tolist()
				output[_feature] = (rawData['Close'] - rolling_max)/rolling_max

			elif _feature[0:3] == 'MIN':
				_window = np.int(_feature[3:])

				min_ = np.int(_window * 0.9)
				rolling_min = pd.Series.rolling(rawData['Close'],window=_window,min_periods=min_).min().tolist()
				output[_feature] = (rawData['Close'] - rolling_min)/rolling_min


		return output	
		
	def get_classification_output(self,rawData,PrizeThresholds=None,duration = None):
		
		'''
		To do make sure that the time difference are indeed 10 BT

		Compute binary output for stock classification 

		Parameters
		---------------
		rawData : string OR pandas DataFrame, path to raw stock data OR raw data

		PrizeThresholds : List of floats , thereshold for classification of stock gain or loss List has to be ascending

		duration : int , number of business days to evalute stock gain/loss

		
		Returns
		---------------
		classifier : pandas DataFrame, binary classifier for stock prize of stock given in path_raw


		Example
		--------------
		To Do....

		'''

		#check for input

		if isinstance(rawData,basestring) == True:
			if os.path.isfile(rawData) == True:

				rawData = pd.read_csv(rawData)

			else:
				raise ValueError('Input path to raw data does not exist')

		
		if PrizeThresholds is None:
			PrizeThresholds = self.PrizeThresholds

		if duration is None:
			duration = self.duration


		classifier = pd.DataFrame()
		classifier['Date'] = pd.Series(rawData['Date'],index=rawData.index[0:-duration])

		
		if np.any((np.sort(PrizeThresholds) == PrizeThresholds) == False):
			raise ValueError('Input parameter "PrizeThresholds" is not ascendingly sorted')



		#create classifier columns from PrizeThresholds

		tmp = np.array((rawData['Close'].values[duration:]- rawData['Close'].values[0:-duration])/rawData['Close'].values[0:-duration])*100.

		
		classifier['<'+str(PrizeThresholds[0])] = pd.Series(np.nan,index=classifier.index)
		
		
		Inds2Keep = np.where(np.isfinite(tmp))[0]
		classifier.loc[Inds2Keep, '<'+str(PrizeThresholds[0])] = tmp[Inds2Keep]<=PrizeThresholds[0]	

		for i in range(1,len(PrizeThresholds)):
			classifier[str(PrizeThresholds[i-1])+'<'+str(PrizeThresholds[i])] = pd.Series(np.nan,index=classifier.index)
			classifier.loc[Inds2Keep,str(PrizeThresholds[i-1])+'<'+str(PrizeThresholds[i])] = np.bitwise_and(tmp[Inds2Keep]>PrizeThresholds[i-1],tmp[Inds2Keep]<=PrizeThresholds[i])

		classifier['>'+str(PrizeThresholds[-1])] = pd.Series(np.nan,index=classifier.index)
		classifier.loc[Inds2Keep,'>'+str(PrizeThresholds[-1])] = tmp[Inds2Keep]>PrizeThresholds[-1]

		return classifier



	def _return_relative_roll_mean(self,raw_data,window_size):

		'''
		compute and return relative prize difference w.r.t. rolling mean of given window size

		'''
		min_ = np.int(window_size * 0.9)

		rolling_mean = pd.Series.rolling(raw_data['Close'],window=window_size,min_periods=min_).mean().tolist()

		return (raw_data['Close'] - rolling_mean)/rolling_mean

	
	def _return_relative_bollinger_bands(self,rawData,window_size=20,k=2):

		'''
		compute relative bollinger bands

		Parameters
		--------------
		rawData : pandas DataFrame

		window_size : int (default 20) window for rolling averages/stds

		k : int (default 2) , number of STD used for band with 

		Returns
		--------------
		upper : np.ndarray relative distance to upper band
		
		lower : np.ndarray relative distance to lower band

		Example:
		-------------
		To Do ...

		'''
		
		min_ = np.int(np.float(window_size) * 0.9)

		rolling_mean = pd.Series.rolling(rawData['Close'],window=window_size,min_periods=min_).mean().values
		rolling_std =  pd.Series.rolling(rawData['Close'],window=window_size,min_periods=min_).std().values

		upper = rolling_mean + k*rolling_std
		upper = (rawData['Close'] - upper)/upper
		
		lower = rolling_mean - k*rolling_std
		lower = (rawData['Close'] -lower)/lower

		return np.array(lower),np.array(upper)