import numpy as np
import pandas as pd


def get_high_low_delta(rawData,windows=1,relative=False):
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

def get_up_down_move(rawData,windows=1):
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
	tmp =get_high_low_delta(rawData,windows=windows,relative=False)

	out = pd.DataFrame(index=rawData.index)
	out['up move'] = (tmp['High'] + tmp['High'].abs())/2.
	out['down move'] = (-tmp['Low'] + tmp['Low'].abs())/2.

	return out


def get_pdm_ndm(rawData,window=14,smooth='EMA'):
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

	tmp = get_up_down_move(rawData)

	out = pd.DataFrame(index=rawData.index)
	out['PDM'] = np.where(tmp['up move']>tmp['down move'],tmp['up move'],0)
	out['NDM'] = np.where(tmp['down move']>tmp['up move'],tmp['down move'],0)

	
	if window>1 and smooth=='EMA':
		out['PDM'] = get_ema(out,window=window,column='PDM')['EMA'+str(window)]
		out['NDM'] = get_ema(out,window=window,column='NDM')['EMA'+str(window)]
	elif window>1 and smooth == 'Wilder':
		out['PDM'+str(window)] = get_wilders_average(out,column='PDM',window=window)
		out['NDM'+str(window)] = get_wilders_average(out,column='NDM',window=window)
	return out

def get_true_range(rawData):
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


def get_pdi_and_ndi(rawData,window,smooth='Wilder'):
	"""
	compute positive directional moving index and negative directional moving index

	Parameters
	-------------
	rawData : pandas DataFrame must include columns 'High', 'Low', 'Close'

	Returns
	-------------   
	"""

	tmp1 = get_pdm_ndm(rawData,window=window,smooth=smooth)
	tmp2 = get_average_true_range(rawData,window=window,smooth=smooth)

	out = pd.DataFrame(index=rawData.index)
	out['PDI'+str(window)] = tmp1['PDM'+str(window)]/tmp2['ATR'+str(window)] * 100.
	out['NDI'+str(window)] = tmp1['NDM'+str(window)]/tmp2['ATR'+str(window)] * 100.
	return out

def get_average_true_range(rawData,window=14,relative=False,smooth='SMMA'):
	"""
	compute average true range (ATR) of stock 
	https://en.wikipedia.org/wiki/Average_true_range
	Parameters
	--------------
	rawData : pandas DataFrame must include columns 'High', 'Low', 'Close'

	window : int window size for smoothed average of true range values

	Returns
	-------------
	out : pandas DataFrame contains column "ATR"+window

	"""

	trueRange = get_true_range(rawData)

	out = pd.DataFrame(index=rawData.index)
	if smooth == 'SMMA':
		if relative == False:
			out['ATR'+str(window)] = get_smma(trueRange,window=window,column='true range')
		elif relative == True:
			out['ATR'+str(window)] = get_smma(trueRange,window=window,column='true range')/rawData['Close']
	elif smooth =='Wilder':
		out['ATR'+str(window)] = get_wilders_average(trueRange,column='true range',window=window)
	return out

def get_directional_movement_index(rawData,window,smooth='Wilder'):
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
	tmp = get_pdi_and_ndi(rawData,window=window,smooth=smooth)

	out = pd.DataFrame(index=rawData.index)
	out['DX'+str(window)] = 100*(tmp['PDI'+str(window)] - tmp['NDI'+str(window)]).abs()/(tmp['PDI'+str(window)] + tmp['NDI'+str(window)])
	return out

def get_adx(rawData,window_adx=14,window_dx = 14):
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

	tmp = get_directional_movement_index(rawData,window=window_dx)

	out = pd.DataFrame(index=rawData.index)

	out['ADX'+str(window_adx)] = np.NaN
	start = np.where(np.isnan(tmp['DX'+str(window_dx)]))[0][-1] +1 +window_adx  

	out['ADX'+str(window_adx)][start-1] = tmp['DX'+str(window_dx)][start-window_adx:start].mean()

	for i in range(start,len(out)):
		out['ADX'+str(window_adx)][i] = (out['ADX'+str(window_adx)][i-1] *13. + tmp['DX'+str(window_dx)][i])/14.

	return out

def get_wilders_average(data,column,window=14):

	out = pd.DataFrame(index=data.index)

	out[column+str(window)] = np.NaN
	out[column+str(window)][window] = data[column][1:window+1].sum()
	for i in range(window+1,len(out)):
		out[column+str(window)][i] = out[column+str(window)][i-1]*(window-1.)/(1.*window) + data[column][i]
	return out


def get_cci(rawData,window=20):
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

		TP_SMA = TP.rolling(window = window,center=False).mean()

		mean_dev = TP.rolling(center=False, window=window).apply(lambda x: np.fabs(x - x.mean()).mean())

		out = pd.DataFrame(index=rawData.index)

		out['CCI'+str(window)] = (TP - TP_SMA)/(.015*mean_dev)

		return out

def get_ema(data,window,column='Close'):

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
	out['EMA'+str(window)] = data[column].ewm(ignore_na=False,span=window,min_periods=window,adjust=True).mean()

	return out

def get_average_slope(data,window=14,relative=True):
		
		if relative == True:			
			return pd.Series.rolling(data.diff(periods=1),window=window).mean()/data		
		else:
			return pd.Series.rolling(data.diff(periods=1),window=window).mean()

def get_average_for_crossing_direction(data,window=5):

		return pd.Series.rolling(data,window=5).mean()
		
def get_williams(rawData,window=14):
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
	lower = rawData['Low'].rolling(window=window,center=False).min()
	upper = rawData['High'].rolling(window=window,center=False).max()

	out['WR'+str(window)] = (upper - rawData['Close'])/(upper-lower) * -100.

	return out

def get_MACD(rawData,fast_window = 12,slow_window=26,signal_window=9):

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
	fast = get_ema(rawData,window=fast_window)['EMA'+str(fast_window)]
	slow = get_ema(rawData,window=slow_window)['EMA'+str(slow_window)]

	out = pd.DataFrame(index=rawData.index)
	out['MACD'] = fast - slow
	out['MACDS'] = get_ema(out,window=signal_window,column='MACD')['EMA'+str(signal_window)]
	out['MACDH'] = (out['MACD'] - out['MACDS'])

	return out

def get_bollinger_bands(rawData,window_size=20,k=2,relative=True):

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

	rolling_mean = pd.Series.rolling(rawData['Close'],window=window_size,min_periods=window_size).mean()
	rolling_std =  pd.Series.rolling(rawData['Close'],window=window_size,min_periods=window_size).std()

	upper = rolling_mean + k*rolling_std
	lower = rolling_mean - k*rolling_std

	out = pd.DataFrame(index=rawData.index)
	if relative == False:
		out['upper'] = upper
		out['middle'] =  rolling_mean
		out['lower'] = lower
		return out

	elif relative == True:
		out['upper'] = (rawData['Close'] - upper)/upper
		out['middle'] = (rawData['Close'] - rolling_mean)/rolling_mean
		out['lower'] = (rawData['Close'] -lower)/lower
		return out

def get_smma(data,window_size,column='Close'):
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
	out['SMMA'+str(window_size)] = data[column].ewm(ignore_na=False,alpha=1./window_size,min_periods=window_size,adjust=True).mean()

	return out


def get_rsi(rawData,window_size=14):
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

	dpm_smma = get_smma(dpm,window_size=window_size)['SMMA'+str(window_size)]
	dnn_smma = get_smma(dnn,window_size=window_size)['SMMA'+str(window_size)]

	out = pd.DataFrame(index=rawData.index)
	out['RSI'+str(window_size)] = 100. - 100./(1. + dpm_smma/dnn_smma)

	return out

def get_PVO(rawData,window_vol1 = 12, window_vol2 = 26, window_signal = 9):

	'''
	comute percentage volume oscillator (PVO)
	http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:percentage_volume_oscillator_pvo

	default options, window 1 ema volume  = 12 , window 2 ema volume = 26, window ema signal line = 9

	interpretation: amplitude, crossing signal line, i.e., positive/negative of PVOH
	'''

	out = pd.DataFrame(index=rawData.index)
	tmp = self._get_ema(rawData,window=window_vol2,column='Volume')

	out['PVO'] = (get_ema(rawData,window=window_vol1,column='Volume')['EMA'+str(window_vol1)] - tmp['EMA'+str(window_vol2)])/tmp['EMA'+str(window_vol2)] * 100.
	out['PVOS']= get_ema(out,window=window_signal,column='PVO')['EMA'+str(window_signal)]
	out['PVOH'] = out['PVO'] - out['PVOS']

	return out


def rolling_mean(rawData,window_size,column='Close',relative=False):

	'''
	compute and return relative prize difference w.r.t. rolling mean of given window size

	'''
	min_ = np.int(window_size * 0.9)

	out = pd.DataFrame(index=rawData.index)
	rolling_mean = rawData[column].rolling(window=window_size,min_periods=min_).mean()
	#rolling_mean = pd.Series.rolling(raw_data['Close'],window=window_size,min_periods=min_).mean()
	#return (raw_data['Close'] - rolling_mean)/rolling_mean
	if relative == True:
		out['SMA'+str(window_size)] = (rawData[column] - rolling_mean)/rolling_mean
		return out
	elif relative == False:
		out['SMA'+str(window_size)] =rolling_mean
		return out