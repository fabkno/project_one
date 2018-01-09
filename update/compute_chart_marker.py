import pandas as pd
import numpy as np
import os
import sys

'''

Compute standard chartmakers, ROLMEAN: 38,50,100,200, Bollinger Bands


'''


def return_relative_roll_mean(raw_data,window_size):

	'''
	compute and return relative prize difference w.r.t. rolling mean of given window size

	'''
	min_ = np.int(window_size * 0.9)

	rolling_mean = pd.Series.rolling(raw_data['Close'],window=window_size,min_periods=min_).mean().tolist()

	return (raw_data['Close'] - rolling_mean)/rolling_mean



def get_chartdata(rawData,ListOfChartFeatures = ['GD200','GD100','GD50','GD38']):

	'''

	Parameters
	-------------
	path_raw : string, path to raw stock data

	ListOfChartFeatures : list of strings (default ['GD200','GD100','GD50','GD38']) correspond to rolling averages


	Returns
	-------------
	
	output : pandas DataFrame 	


	'''
	
	#check for input

	if isinstance(rawData,basestring) == True:
		if os.path.isfile(rawData) == True:
			rawData = pd.read_csv(rawData)
		else:
			raise ValueError('Input path to raw data does not exist')

	
	output = pd.DataFrame()
	output['Date'] = pd.Series(rawData['Date'],index = rawData.index)
	output['Close'] = pd.Series(rawData['Close'],index=rawData.index)

	for _feature in ListOfChartFeatures:

		if _feature[0:2] == 'GD':	

			output[_feature] = pd.Series(return_relative_roll_mean(rawData,np.int(_feature[2:])),index=rawData.index)

	#GD = ListOfChartFeatures[np.argmax([int(_feature[2:]) for _feature in ListOfChartFeatures])]
	
	#nonZeroInds = output.loc[output[GD].notnull()].index.tolist()

	return output


def get_classification_output(rawData,PrizeThresholds=[-5,-2.5,0,2.5,5],duration = 10):
	
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

	
	classifier = pd.DataFrame()
	classifier['Date'] = pd.Series(rawData['Date'],index=rawData.index[0:-duration],copy=True)

	
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




def update_chart_data():

 	'''

 	Compute and update chart tool data

 	'''
 	ListOfCountries = ['Germany']
 	current_path = os.getcwd()
 	parent_path = os.path.dirname(os.getcwd())

 	for _country in ListOfCountries:
 		os.chdir(current_path)
	
 	 	path_raw = parent_path+'/data/raw/'+_country+'/'
 	 	path_chart = parent_path+'/data/chart/'+_country+'/'

 		ListOfIndices = next(os.walk(path_raw))[1]

 		for _index in ListOfIndices:

 			if os.path.exists(path_chart+_index) == False:
 				os.makedirs(path_chart+_index)
			

 			os.chdir(path_raw+_index)

 	 		if os.path.isfile('ListOfCompanies.csv'):
  				labels = pd.read_csv('ListOfCompanies.csv')['Label']

  				if len(labels) >=1:
  					for _label in labels:
  						if os.path.isfile(_label+'.csv'):
 	 						rawData = pd.read_csv(_label+'.csv')

 		 					InputData = get_chartdata(rawData)
 		 					OutputData = get_classification_output(rawData)

 		 					#find index with NAN
 		 					indsInput = InputData.loc[InputData.notnull().any(axis=1)].index.tolist()
 		 					indsOutput = OutputData.loc[OutputData.notnull().any(axis=1)].index.tolist()
 		 					
 		 					#find intersection betwen both index lists
		 					inds_final = list(set(indsInput) & set(indsOutput))

		 					#write final data
 		 					InputData.loc[inds_final].to_csv(path_chart+_index+'/'+_label+'_input.csv')
 		 					OutputData.loc[inds_final].to_csv(path_chart+_index+'/'+_label+'_output.csv')

 		 					print "chart values for ", _label, " written"


 	 			else:
 					print "Searched index does not have any entries"
 			else:
 					print country, ": Index : ",index, " not found"

 			print "########## Index ", _index, " successfully updated #########\n\n"

update_chart_data()
