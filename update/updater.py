import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import os
from pandas_datareader._utils import RemoteDataError
import stockstats

class Updater(object):
	"""

	Add description

	"""
	def __init__(self,PathData=None):

		self.ListOfCountries = ['Germany','US']
		self.ListOfIndices = {'Germany':['DAX','MDAX','SDAX'],'US':['DOW']}
		
		'''
		ListOfChartFeatures: "GDXX" = moving average and number of days
							 "BB_20_2" = bolling bands (tau = 20 and k=2)
							 "RSI_14" = relative strength index (last 14 days)
							 "ADX" = average directional index
							 "MACD" = moving average convergence/divergence ###check number of days
		'''
		self.ListOfChartFeatures = ['GD200','GD100','GD50','GD38','BB_20_2','RSI_14','ADX','MACD','MAX20','MAX65','MAX130','MAX260','MIN20','MIN65','MIN130','MIN260']

		'''
		PrizeThresholds : threshold to ategorize relative (in percent) stock evolution within N days

		'''
		self.PrizeThresholds=[-5,-2.5,0,2.5,5]
		self.duration = 10 #duration for which to compute classification 

		self.UpdateTimeEnd = datetime.datetime.today()
		self.UpdateTimeStart = datetime.datetime(2009,1,1)

		if PathData is None:
			self.PathData = os.path.dirname(os.getcwd())+'/data/'
		else:
			self.PathData = PathData
		

	def update_all(self):

		self.update_stock_prizes()
		self.update_chart_markers_and_output()


	def update_stock_prizes(self,ListOfIndices=None):

		"""
		Update stock prizes using yahoo finance 

		Paramters
		--------------

		ListOfIndices : dictionary (default None), dict keys are countries and its values are list of stock indices.
						 If None provided use the full country list defined in the class constructor

		"""

		if ListOfIndices is None:
			ListOfIndices = self.ListOfIndices

		for _country in ListOfIndices.keys():

			for _StockIndex in ListOfIndices[_country]:
				IndexPath = self.PathData+'raw/'+_country+'/'+_StockIndex+'/'
				if os.path.isfile(IndexPath+'ListOfCompanies.csv'):
					
					labels = pd.read_csv(IndexPath+'ListOfCompanies.csv')['Label']
					
					if len(labels) >1:
						for _label in labels:
							
							try:
								stock_prize = pdr.get_data_yahoo(_label,self.UpdateTimeStart,self.UpdateTimeEnd)
								stock_prize.to_csv(IndexPath +_label+'.csv')
								
								print "Stock ",_label, " updated"

							except RemoteDataError:
								print "No information for ticker ", _label
								continue

					else:	print "Searched index does not have any entries"
				
				else:	print _country, ": Index : ",_StockIndex, " not found"

				print "\n########## Index ", _StockIndex, " successfully updated #########\n\n"

	def update_chart_markers_and_output(self,ListOfIndices=None):

		'''	
		update chart indicators and classification output from rawdata

		
		'''

		if ListOfIndices is None:
			ListOfIndices = self.ListOfIndices


		for _country in ListOfIndices.keys():

			for _StockIndex in ListOfIndices[_country]:
				
				IndexPathRaw = self.PathData+'raw/'+_country+'/'+_StockIndex+'/'
				IndexPathChart = self.PathData+'chart/'+_country+'/'+_StockIndex+'/'

				if os.path.exists(IndexPathChart) is False:
					os.makedirs(IndexPathChart)

				
				if os.path.isfile(IndexPathRaw+'ListOfCompanies.csv'):
					
					labels = pd.read_csv(IndexPathRaw+'ListOfCompanies.csv')['Label']
					
					if len(labels) >1:
						for _label in labels:
							if os.path.isfile(IndexPathRaw+'/'+_label+'.csv'):
								rawData = pd.read_csv(IndexPathRaw+'/'+_label+'.csv')

	 		 					InputData = self.get_chartdata(rawData)
	 		 					OutputData = self.get_classification_output(rawData)

	 		 					#find index with NAN
	 		 					indsInput = InputData.loc[InputData.notnull().all(axis=1)].index.tolist()
	 		 					indsOutput = OutputData.loc[OutputData.notnull().all(axis=1)].index.tolist()
	 		 					
	 		 					#find intersection betwen both index lists
			 					inds_final = list(set(indsInput) & set(indsOutput))

			 					
			 					#write final data
	 		 					InputData.loc[inds_final].to_csv(IndexPathChart+_label+'_input.csv')
	 		 					OutputData.loc[inds_final].to_csv(IndexPathChart+_label+'_output.csv')

	 		 					print "chart values for ", _label, " written"




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
			else:
				raise ValueError('Input path to raw data does not exist')

		
		output = pd.DataFrame()
		output['Date'] = pd.Series(rawData['Date'],index = rawData.index)
		output['Close'] = pd.Series(rawData['Close'],index=rawData.index)

		###change eventually by own implementation

		tmp = stockstats.StockDataFrame.retype(rawData.copy())


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
				output[_feature] = tmp.get('rsi_'+str(_window)).values
			elif _feature[0:3] == 'ADX':
				output[_feature] = tmp.get('adx').values

			elif _feature[0:4] == 'MACD':
				output[_feature] = tmp.get('macd').values

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

		tmp = np.array((rawData['Close'].values[0:-duration] - rawData['Close'].values[duration:])/rawData['Close'].values[0:-duration])*100.

		
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