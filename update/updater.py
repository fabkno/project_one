import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import os,shutil,sys
from pandas_datareader._utils import RemoteDataError
import stockstats
import chart_tools as ct
from logger import Log

	# 		#find index with NAN
	# 		indsInput = InputData.loc[InputData.notnull().all(axis=1)].index.tolist()
	# 		indsOutput = OutputData.loc[OutputData.notnull().all(axis=1)].index.tolist()
			
	# 		#find intersection betwen both index lists
	# 		inds_final = list(set(indsInput) & set(indsOutput))

class StockUpdater(Log):
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
		

		Log.__init__(self,PathData=PathData)

		self.ListOfChartFeatures = ['GD200','GD100','GD50','GD38','BB_20_2','RSI7','RSI14','RSI25','WR14','CCI20','ADX','MACD','MAX20','MAX65','MAX130','MAX260','MIN20','MIN65','MIN130','MIN260']

		'''
		PrizeThresholds : threshold to categorize relative (in percent) stock evolution within N days

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

	def update_stock_prizes(self,ListOfTickers =None):

		"""
		Update stock prizes given in ListOfCompanies using yahoo finance 

		If there is data to update the old file is backuped to .../backup/stockTicker.p so the backup is good for one business day

		"""

		if ListOfTickers is None:
			ListOfTickers = self.ListOfCompanies['Yahoo Ticker']

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
					self.logging("Stock "+stocklabel+": UpdateTimeStart is equal to UpdateTimeEnd ")
					continue
				try:
					
					stock_prize = pdr.get_data_yahoo(stocklabel,self.UpdateTimeStart,self.UpdateTimeEnd)
					stock_prize.dropna(inplace=True)
					stock_prize.drop(index=stock_prize.loc[stock_prize['Volume'] == 0.0].index.tolist(),inplace=True)
					stock_prize.reset_index(inplace=True)

					#print stock_prize
					stock_prize = stock_prize.loc[stock_prize['Date']>self.UpdateTimeStart]
					
					if len(stock_prize) == 0:
						self.logging("Stock "+stocklabel+": no new data available")
						continue
					
					StockValue = pd.concat([StockValue, stock_prize], ignore_index=True)

					shutil.copy(self.PathData+'raw/stocks/'+stocklabel+'.p',self.PathData+'raw/stocks/backup/'+stocklabel+'.p')
					
					#print "number of rows", len(StockValue), " for label", stocklabel
					StockValue.reset_index(inplace=True,drop=True)
					
					StockValue.to_pickle(self.PathData+'raw/stocks/'+stocklabel+'.p')
					print "Stock ",stocklabel, " updated"
					self.logging("Stock "+stocklabel+": successfully updated")

				except RemoteDataError:
					self.logging("Stock "+stocklabel+": No information for ticker found")
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
					self.logging("Stock "+stocklabel+": successfully updated")

				except RemoteDataError:
					self.logging("Stock "+stocklabel+": No information for ticker found")
					print "No information for ticker ", stocklabel
					continue

		print "\nFinished updating stock prizes\n\n"


	def update_chart_markers(self,ListOfTickers = None):
		'''	
		update chart indicators from raw chart data
		st
		'''		

		if ListOfTickers is None:
			ListOfTickers = self.ListOfCompanies['Yahoo Ticker']

		print "Start updating chart markers"
		print "--------------------------------\n"
		for stocklabel in ListOfTickers:
			
			#check if raw stock data exists
			if os.path.isfile(self.PathData + 'raw/stocks/'+stocklabel+'.p'):
				rawData = pd.read_pickle(self.PathData + 'raw/stocks/'+stocklabel+'.p')	
				ChartData = self.get_chartdata(rawData)
				ChartData.dropna(inplace=True)

				ChartData.to_pickle(self.PathData+'chart/stocks/'+stocklabel+'.p')
				self.logging("Stock "+stocklabel+": chart values successfully written")
				print "chart values for ", stocklabel, " written"
			else: 
				self.logging("Stock "+stocklabel+":raw data does not exist")
				print "raw stock data for stock ",stocklabel, " does not exist"
				print "try running update_stock_prize() first"
		print "\nFinished updating chart markers\n\n"
	
	def update_stock_classification(self,ListOfTickers = None):

		'''
		update stock classification 

		'''
		if ListOfTickers is None:
			ListOfTickers = self.ListOfCompanies['Yahoo Ticker']

		print "Start updating stock classification"
		print "--------------------------------------\n"
		for stocklabel in self.ListOfCompanies['Yahoo Ticker']:
			
			#check if raw stock data exists
			if os.path.isfile(self.PathData + 'raw/stocks/'+stocklabel+'.p'):
				rawData = pd.read_pickle(self.PathData + 'raw/stocks/'+stocklabel+'.p')	
				classification = self.get_classification_output(rawData)
				classification.dropna(inplace=True)

				classification.to_pickle(self.PathData+'classification/stocks/'+stocklabel+'.p')
				self.logging("Stock "+stocklabel+": classification values successfully written")
				print "classification values for ", stocklabel, " written"
			else: 
				self.logging("Stock "+stocklabel+":raw data does not exist")
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
					self.logging("ValueError: Critical!!!! >>rawData<< contains duplicated Dates. Classification output is most probably inaccurate ")
					raise ValueError('Critical!!!! "rawData" contains duplicated Dates. Classification output is most probably inaccurate')
			else:
				self.logging("ValueError: Input path for raw data does not exist")
				raise ValueError('Input path for raw data does not exist')

		
		output = pd.DataFrame(index=rawData.index)
		output['Date'] = rawData['Date']
		output['Close'] = rawData['Close']
		output['Volume'] = rawData['Volume']


		###change eventually by own implementation

		#tmp = stockstats.StockDataFrame.retype(rawData.copy(deep=True))


		for _feature in ListOfChartFeatures:
			
			if _feature[0:2] == 'GD':	
				out = ct.rolling_mean(rawData,window_size=np.int(_feature[2:]),column='Close')
				output[_feature] = out['SMA'+str(_feature[2:])]
				output[_feature+'X'] =ct.get_average_for_crossing_direction(out['SMA'+str(_feature[2:])])
			
			#evtl. do slope of GD
			#elif _feature[0:5] == 'DOTGD':
			#	_window = np.int(_feature[5:])
			#	min_ = np.int(_window * 0.9)

				#rolling_mean  = pd.Series.roling(rawData['Close'],window=_window)
			elif _feature[0:2] == 'BB':
				_k = np.int(_feature[-1])
				_window = np.int(_feature[3:[i for i,x in enumerate(_feature) if x=='_'][1]])

				_out = ct.get_bollinger_bands(rawData,window_size=_window,k=_k,relative=True)

				#if len(lower) != len(output):
				#	self.logging("ValueError: Caution length of BB bands not equal length of dates")
				#	raise ValueError('Caution length of BB bands not equal length of dates')

				output['Lower_'+_feature] = _out['lower']
				output['Upper_'+_feature] = _out['upper']
				output['Middle_'+_feature] = _out['middle']

				#output['LowerX_'+_feature] = self._get_average_for_crossing_direction(_out['lower'],window = 3)
				#output['UpperX_'+_feature] = self._get_average_for_crossing_direction(_out['upper'],window = 3)
				#output['MiddleX_'+_feature] = self._get_average_for_crossing_direction(_out['middle'],window = 3)


			elif _feature[0:3] == 'CCI':
				_window = np.int(_feature[3:])
				_out = ct.get_cci(rawData,window=_window)
				output[_feature] = _out['CCI'+str(_window)]
				output[_feature+'X'] = ct.get_average_for_crossing_direction(_out['CCI'+str(_window)])


			elif _feature[0:2] == 'WR':
				_window = np.int(_feature[2:])
				_out = ct.get_williams(rawData,window=_window)
				output[_feature] = _out['WR'+str(_window)]
				output[_feature+'X'] = ct.get_average_for_crossing_direction(_out['WR'+str(_window)])

			elif _feature[0:3] == 'RSI':

				_window = np.int(_feature[3:])
				#values = tmp.get('rsi_'+str(_window)).values		
				_out = ct.get_rsi(rawData,window_size=_window)
				output[_feature] =_out['RSI'+str(_window)]
				output[_feature+'X'] = ct.get_average_for_crossing_direction(_out['RSI'+str(_window)])

			elif _feature[0:3] == 'ADX':
				_out = ct.get_adx(rawData,window_dx=14,window_adx=14)	
				_out2 = ct.get_pdi_and_ndi(rawData,window=14,smooth='Wilder')

				output['PDI14R'] =_out['ADX14'] - _out2['PDI14']
				output['NDI14R'] =_out['ADX14'] - _out2['NDI14']

				output['PDI14RX'] = ct.get_average_for_crossing_direction(output['PDI14R'],window=2)
				output['NDI14RX'] = ct.get_average_for_crossing_direction(output['NDI14R'],window=2)
				#_out2 = self._get

				#output[_feature] = pd.Series(tmp.get('adx').values,index=rawData.index)

			elif _feature[0:4] == 'MACD':
				_out = ct.get_MACD(rawData)

				output[_feature] = _out['MACD']
				output[_feature+'X'] = ct.get_average_for_crossing_direction(_out['MACD'])

				output[_feature+'H'] = _out['MACDH']
				output['MACDHX'] = ct.get_average_for_crossing_direction(_out['MACDH'])
				
				#output[_feature] = pd.Series(tmp.get('macd').values,index=rawData.index)

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
				self.logging("ValueError: Input path for raw data does not exist")
				raise ValueError('Input path to raw data does not exist')

		
		if PrizeThresholds is None:
			PrizeThresholds = self.PrizeThresholds

		if duration is None:
			duration = self.duration


		classifier = pd.DataFrame()
		classifier['Date'] = pd.Series(rawData['Date'],index=rawData.index[0:-duration])

		
		if np.any((np.sort(PrizeThresholds) == PrizeThresholds) == False):
			self.logging("ValueError: Input parameter PrizeThresholds is not ascendingly sorted")
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



	def _return_relative_roll_mean(self,rawData,window_size,column='Close'):

		'''
		compute and return relative prize difference w.r.t. rolling mean of given window size

		'''
		min_ = np.int(window_size * 0.9)

		out = pd.DataFrame(index=rawData.index)
		rolling_mean = rawData[column].rolling(window=window_size,min_periods=min_).mean()
		#rolling_mean = pd.Series.rolling(raw_data['Close'],window=window_size,min_periods=min_).mean()
		#return (raw_data['Close'] - rolling_mean)/rolling_mean
		out['SMA'+str(window_size)] = (rawData[column] - rolling_mean)/rolling_mean
		return out

	
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
		upper : relative distance to upper band
		
		middle : relative distance to middle band

		lower : relative distance to lower band

		interpretation: amplitude, crossing lower, upper, middle, W and M pattern 

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

		out = pd.DataFrame(index=rawData.index)
		out['upper'] = upper
		out['middle'] = (rawData['Close'] - rolling_mean)/rolling_mean
		out['lower'] = lower

		return out


	def _get_smma(self,data,window_size,column='Close'):
		'''
		compute smoothed exponentially weighted average

		Parameters
		--------------
		rawData : pandas data frame must include closing prize

		window_size : int window for weigthed average

		Returns
		--------------
		pandas DataFrame 

		'''
		out = pd.DataFrame(index=data.index)
		out['SMMA'+str(window_size)] = data[column].ewm(ignore_na=False,alpha=1./window_size,min_periods=0,adjust=True).mean()

		return out

	def _get_rsi(self,rawData,window_size=14):
		'''
		compute relative strength index for given window_size 

		Parameters
		--------------
		rawData : pandas data frame must include closing prize

		window_size : int number of days

		column : string (default ='Close'), idicates column for average computation

		Returns
		--------------
		pandas DataFrame 

		Interpretation
		-------------------
		amplitude, crossing, divergence 

		'''
		diff = rawData['Close'].diff(periods=1)
		dpm = pd.DataFrame()
		dnn = pd.DataFrame()
		dpm['Close'] = (diff+diff.abs())/2.
		dnn['Close'] = (-diff+diff.abs())/2.

		dpm_smma = self._get_smma(dpm,window_size=window_size)['SMMA'+str(window_size)]
		dnn_smma = self._get_smma(dnn,window_size=window_size)['SMMA'+str(window_size)]

		out = pd.DataFrame(index=rawData.index)
		out['RSI'+str(window_size)] = 100. - 100./(1. + dpm_smma/dnn_smma)

		return out


	def _get_ema(self,data,window,column='Close'):

		'''
		compute exponentially weighted moving average

		Parameters
		-------------
		rawData : pandas DataFrame

		window_size : int window for moving average

		column : string (default ='Close'), idicates column for average computation

		Returns
		-------------
		pandas DataFrame

		'''
		out = pd.DataFrame(index=data.index)
		out['EMA'+str(window)] = data[column].ewm(ignore_na=False,span=window,min_periods=0,adjust=True).mean()

		return out

	def _get_MACD(self,rawData,fast_window = 12,slow_window=26,signal_window=9):

		'''
		computation of moving average convergence/divergence

		interpretation: MACD amplitude, crossing, MACDH amplitude crossing, MACD divergence, i.e., reverse trends of local to global minimum/maximum between MACD and Close

		Parameters
		---------------

		rawData : pandas DataFrame must include 'Close' prize

		fast_window : int (default 12),fast window for exp. weighted moving average of closing prize

		slow_window : int (default 26), slow window for exp. weighted moving average of closing prize 

		signal_window : int (default 9), window for exp. weighted moving average of MACD 

		Returns
		---------------
		out : pandas DataFrame contains columns 'MACD' = absolute MACD
												'MACDS' = signal line of MACD
												'MACDH' = relative (corrected) MACD

		'''
		fast = self._get_ema(rawData,window=fast_window)['EMA'+str(fast_window)]
		slow = self._get_ema(rawData,window=slow_window)['EMA'+str(slow_window)]

		out = pd.DataFrame(index=rawData.index)
		out['MACD'] = fast - slow
		out['MACDS'] = self._get_ema(out,window=signal_window,column='MACD')['EMA'+str(signal_window)]
		out['MACDH'] = (out['MACD'] - out['MACDS'])

		return out
	def _get_high_low_delta(self,rawData,windows=1,relative=False):
		'''
		get difference between daily high and low values within window range, respectively
		        
		Parameters
		---------------

		rawData : pandas DataFrame must include daily 'High' and 'Low' prize

		windows : int (default 1), window for taking the difference, e.g. window=2 takes the differences between every 2nd day

		relative : bool (default False) whether or not return the relative (or absolute) difference

		Returns
		----------------
		    
		out : pandas DataFrame contains high and low values (relative or absolute)

		'''
		if 'High' not in rawData.columns:
			raise ValueError('rawData does not contain column "High"')
		if 'Low' not in rawData.columns:
			raise ValueError('rawData does not contain column "Low"')

		out = pd.DataFrame(index=rawData.index)
		if relative == False:
			out['High'] = rawData['High'].diff(periods=windows)
			out['Low'] = rawData['Low'].diff(periods=windows)
		elif relative == True:
			out['High rel'] = rawData['High'].diff(periods=windows)/rawData['High']
			out['Low rel'] = rawData['Low'].diff(periods=windows)/rawData['Low']

		return out
  
	def _get_up_down_move(self,rawData,windows=1):
		'''
		To do

		Parameters
		-------------
		rawData : pandas DataFrame must include daily 'High' and 'Low' prize

		windows : int (default 1) window for getting the respective up and down moves of stock prize

		Returns
		-------------
		out : pandas DataFrame contains columns "up move" and "down move" for both moves respectively
		'''
		tmp =self._get_high_low_delta(rawData,windows=windows,relative=False)

		out = pd.DataFrame(index=rawData.index)
		out['up move'] = (tmp['High'] + tmp['High'].abs())/2.
		out['down move'] = (-tmp['Low'] + tmp['Low'].abs())/2.

		return out


	def _get_pdm_ndm(self,rawData,window):
		'''
		compute positive and negative directional moving average (negative directional moving accumulation)

		Parameters
		-------------
		rawData : pandas DataFrame must include daily 'High' and 'Low' prize

		window : int number of business days for computation of moving average

		Returns
		-------------
		out : pandas DataFrame contains columns "pdm" and "ndm" for positve (negative) directional moving average

		'''

		tmp = self._get_up_down_move(rawData)

		out = pd.DataFrame(index=rawData.index)
		out['PDM'] = np.where(tmp['up move']>tmp['down move'],tmp['up move'],0)
		out['NDM'] = np.where(tmp['down move']>tmp['up move'],tmp['down move'],0)

		if window>1:
			out['PDM'] = self._get_ema(out,window=window,column='PDM')['EMA'+str(window)]
			out['NDM'] = self._get_ema(out,window=window,column='NDM')['EMA'+str(window)]

		return out
	
	def _get_pdi_and_ndi(self,rawData,window):
		"""
		compute positive directional moving index and negative directional moving index

		Parameters
		-------------
		rawData : pandas DataFrame must include columns 'High', 'Low', 'Close'

		window : int window size

		Returns
		-------------   
		out : pandas DataFrame with columns "PDI" (positive directional index) and "NDI" (negative directional index)
		
		"""

		tmp1 = self._get_pdm_ndm(rawData,window=window)
		tmp2 = self._get_average_true_range(rawData,window=window)

		out = pd.DataFrame(index=rawData.index)
		out['PDI'+str(window)] = tmp1['PDM']/tmp2['ATR'+str(window)] * 100.
		out['NDI'+str(window)] = tmp1['NDM']/tmp2['ATR'+str(window)] * 100.
		return out


	def _get_true_range(self,rawData):
		"""
		compute true range

		Parameters
		-------------
		rawData : pandas DataFrame must include columns 'High', 'Low', 'Close'

		Returns
		-------------
		out : pandas DataFrame contains column "true range" 
		"""

		prev_close = rawData['Close'].shift(periods=1)

		c1 = rawData['High'] - rawData['Low']
		c2 = np.abs(rawData['High'] - prev_close)
		c3 = np.abs(rawData['Low'] - prev_close)

		out = pd.DataFrame(index=rawData.index)
		out['true range'] = np.max((c1,c2,c3),axis=0)
		return out

	def _get_average_true_range(self,rawData,window=14,relative=False):
		"""
		compute average true range (ATR) of stock 
		https://en.wikipedia.org/wiki/Average_true_range

		Parameters
		--------------
		rawData : pandas DataFrame must include columns 'High', 'Low', 'Close'

		window : int window size for smoothed average of true range values

		relative : bool (default False), when true ATR is divided by closing stock prize

		Returns
		-------------
		out : pandas DataFrame contains column "ATR"+window

		"""

		trueRange = self._get_true_range(rawData)

		out = pd.DataFrame(index=rawData.index)
		if relative == False:
			out['ATR'+str(window)] = self._get_smma(trueRange,window_size=window,column='true range')['SMMA'+str(window)]
		elif relative == True:
			out['ATR'+str(window)] = self._get_smma(trueRange,window_size=window,column='true range')['SMMA'+str(window)]/rawData['Close']

		return out

	def _get_directional_movement_index(self,rawData,window):
		'''
		compute directional movement index (dx)

		Parameters
		------------
		rawData : pandas DataFrame must include columns 'High', 'Low', 'Close'

		window : int window size to compute directional movement index

		Returns
		------------
		out : pandas DataFrame 

		'''
		tmp = self._get_pdi_and_ndi(rawData,window=window)

		out = pd.DataFrame(index=rawData.index)
		out['DX'+str(window)] = 100*(tmp['PDI'+str(window)] - tmp['NDI'+str(window)]).abs()/(tmp['PDI'+str(window)] + tmp['NDI'+str(window)])
		return out

	def _get_raw_stochastic_value(self,rawData,window):
		'''
		compute raw stochastic value for given window

		'''
		low_min = rawData['Low'].rolling(min_periods=1,window=window,center=False).min()
		high_max = rawData['High'].rolling(min_periods=1,window=window,center=False).max()

		out = pd.DataFrame(index=rawData.index)
		out['RSV'+str(window)] = (rawData['Close'] - low_min) /(high_max - low_min) * 100
		#out['RSV'+str(window)].fillna(0).astype('float64')

		return out


	def _get_adx(self,rawData,window_adx=6,window_dx = 14):
		'''
		compute averaged directional movement index

		Parameters
		------------
		rawData : pandas DataFrame must include columns 'High', 'Low', 'Close'

		window_dx : int (default 14) window size to compute directional movement index

		window_adx : int (default 6) window size to compute averaged dx

		Returns
		-------------
		out : pandas DataFrame

		'''

		tmp = self._get_directional_movement_index(rawData,window=window_dx)

		out = pd.DataFrame(index=rawData.index)
		out['ADX'+str(window_adx)+'_'+str(window_dx)] = self._get_ema(tmp,window=window_adx,column='DX'+str(window_dx))['EMA'+str(window_adx)]
		out['ADXR'] = self._get_ema(out,window=window_adx,column='ADX'+str(window_adx)+'_'+str(window_dx))['EMA'+str(window_adx)]
		return out

	def _get_cci(self,rawData,window=20):
		'''
		comute commodity channel index 

		Parameters
		-------------
		rawData : pandas DataFrame must include columns 'High', 'Low', 'Close'

		window : int (default 20) window size to compute necessary rolling averages

		Returns
		-------------

		out : pandas DataFrame

		Interpretation
		------------------

		amplitude, crossing
		'''

		TP = (rawData['Close'] + rawData['High'] + rawData['Low']) / 3.0

		TP_SMA = TP.rolling(min_periods=1,window = window,center=False).mean()

		mean_dev = TP.rolling(min_periods=1, center=False, window=window).apply(lambda x: np.fabs(x - x.mean()).mean())

		out = pd.DataFrame(index=rawData.index)

		out['CCI'+str(window)] = (TP - TP_SMA)/(.015*mean_dev)

		return out




	def _get_trix(self,data,window_trix=15,window_trix_ema = 9,column='Close'):
		'''
		compute triple expanentially weighted graph of data

		Parameters
		----------------
		data : pandas DataFrame 

		window_trix : int (default 15) window for doing the triple average

		window_trix_ema : int (default 9) window for getting signal line of trix

		column : string (default 'Close') column of data for trix is computed

		Returns
		----------------
		out : pandas DataFrame with columns 'TRIX'+window_trix , 'TRIX'+window_trix+_'EMA'+window_ema corresponds to signal line, 'TRIXH', difference between trix and its signal line
	

		'''
		single = self._get_ema(rawData=data,window=window_trix,column=column)
		double = self._get_ema(rawData=single,window=window_trix,column='EMA'+str(window_trix))
		triple = self._get_ema(rawData=double,window=window_trix,column='EMA'+str(window_trix))

		prev_triple = triple.shift(periods=1)

		out = pd.DataFrame(index=data.index)
		out['TRIX'+str(window_trix)] = (triple - prev_triple) * 100. /prev_triple
		out['TRIX'+str(window_trix)+'_EMA'+str(window_trix_ema)] = self._get_ema(out,window=window_trix_ema,column='TRIX'+str(window_trix))
		out['TRIXH'] = out['TRIX'+str(window_trix)] - out['TRIX'+str(window_trix)+'_EMA'+str(window_trix_ema)]

		return out

	def _get_williams(self,rawData,window=14):
		'''
		compute william momentum indicator

		optional try window = 125

		interpretation : crossing, 

		Parameters
		------------
		rawData : pandas DataFrame

		window : int (default 14) window size for computing rolling min/max
		'''
		out = pd.DataFrame(index=rawData.index)
		lower = rawData['Low'].rolling(min_periods=1,window=window,center=False).min()
		upper = rawData['High'].rolling(min_periods=1,window=window,center=False).max()

		out['WR'+str(window)] = (upper - rawData['Close'])/(upper-lower) * 100.

		return out



	def _get_kd(self,data,column):
		'''
		compute ,..
		'''
		p0 = 2./3.
		p1 = 1./3.
		k = 50.
		for i in p1 * data[column].values:
			k = p0 * k +i
			yield k
    
	def _get_kdjk(self,rawData,window):
		'''
		to do
		'''

		tmp = self._get_raw_stochastic_value(rawData,window=window)
		out = pd.DataFrame(index=tmp.index)
		test = np.array(list(self._get_kd(tmp,column='RSV'+str(window))))
		out['KDJK'+str(window)] =test

		return out

	def _get_kdjd(self,rawData,window):
		'''
		to do
		'''
		tmp = self._get_kdjk(rawData,window)
		out = pd.DataFrame(index=tmp.index)
		out['KDJK'+str(window)] = tmp['KDJK'+str(window)]
		out['KDJD'+str(window)] = list(self._get_kd(tmp,column='KDJK'+str(window)))

		return out

	def _get_kdjj(self,rawData,window):
		'''
		compute stochastic oscillator
		'''
		tmp = self._get_kdjd(rawData,window)
		out = pd.DataFrame(index=tmp.index)
		out['KDJK'+str(window)] = tmp['KDJK'+str(window)]
		out['KDJD'+str(window)] = tmp['KDJD'+str(window)]
		out['KDJJ'+str(window)] = 3. *  out['KDJK'+str(window)] - 2.* out['KDJD'+str(window)]
		return out

	def _get_volume_ratio(self,rawData,window=26):
		'''
		compute Volatility Volume Ratio

		'''
		out = pd.DataFrame(index=rawData.index)

		change = rawData['Close'].pct_change() * 100
		out['av'] = np.where(change>0,rawData['Volume'],0)
		out['avs'] = out['av'].rolling(min_periods=1,window=window,center=False).sum() 

		out['bv'] = np.where(change<0,rawData['Volume'],0)
		out['bvs'] = out['bv'].rolling(min_periods=1,window=window,center=False).sum() 

		out['cv'] = np.where(change==0,rawData['Volume'],0)
		out['cvs'] = out['cv'].rolling(min_periods=1,window=window,center=False).sum() 

		out['VR'+str(window)] = (out['avs'] + out['cvs']/2.) / ( out['bvs'] + out['cvs']/2.) *100.

		del out['av']
		del out['bv']
		del out['cv']
		del out['avs']
		del out['bvs']
		del out['cvs']
		return out

	def _get_PVO(self,rawData,window_vol1 = 12, window_vol2 = 26, window_signal = 9):

		'''
		comute percentage volume oscillator (PVO)
		http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:percentage_volume_oscillator_pvo
		
		default options, window 1 ema volume  = 12 , window 2 ema volume = 26, window ema signal line = 9

		interpretation: amplitude, crossing signal line, i.e., positive/negative of PVOH
		'''

		out = pd.DataFrame(index=rawData.index)
		tmp = self._get_ema(rawData,window=window_vol2,column='Volume')

		out['PVO'] = (self._get_ema(rawData,window=window_vol1,column='Volume')['EMA'+str(window_vol1)] - tmp['EMA'+str(window_vol2)])/tmp['EMA'+str(window_vol2)] * 100.
		out['PVOS']= self._get_ema(out,window=window_signal,column='PVO')['EMA'+str(window_signal)]
		out['PVOH'] = out['PVO'] - out['PVOS']

		return out