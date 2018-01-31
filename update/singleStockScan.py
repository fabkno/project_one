import numpy as np
import pandas as pd
import os
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV
import datetime,sys
import util
from pandas.tseries.offsets import BDay

class ModelPrediction(object):

	def __init__(self,FileNameListOfCompanies=None,PathData=None):

		if PathData is None:
				self.PathData = os.path.dirname(os.getcwd())+'/data/'

		else:
			self.PathData = PathData

		if FileNameListOfCompanies is None:
			self.FileNameListOfCompanies ='full_list.csv'
		
		else:
			if os.path.isfile(self.PathData+'company_lists/'+FileNameListOfCompanies) is True:
				self.FileNameListOfCompanies = FileNameListOfCompanies
			else:
				raise ValueError('List: '+FileNameListOfCompanies + ' does not exists in' +self.PathData + 'company_lists/')

		self.ListOfCompanies = pd.read_csv(self.PathData+'company_lists/'+self.FileNameListOfCompanies,index_col='Unnamed: 0')

		if os.path.exists(self.PathData + 'predictions/stocks/') is False:
			os.makedirs(self.PathData + 'predictions/stocks/')

		if os.path.isfile(self.PathData+'predictions/predictions_scan.p') is True:
			self.ListOfPredictions = pd.read_pickle(self.PathData+'predictions/predictions_scan.p')

		else:
			self.ListOfPredictions = None

		if os.path.exists(self.PathData + 'models/stocks/') is False:
			os.makedirs(self.PathData + 'models/stocks/')


	def PredictStocksAll(self,DayOfPrediction=None):

		if DayOfPrediction is None:
			DayOfPrediction = datetime.datetime.today().date()

		predictions = self.PredictStocks(ListOfCompanies=self.ListOfCompanies,DayOfPrediction = DayOfPrediction)

		predictions.to_pickle(self.PathData+'predictions/stocks/daily_predictions_'+str(DayOfPrediction.year)+'_'+str(DayOfPrediction.month)+'_'+str(DayOfPrediction.day))

		print "Daily stock prediction updated"
	def PredictStocks(self,ListOfCompanies,DayOfPrediction):
		'''
		write description

		'''

		try: 
			DayOfPrediction = DayOfPrediction.date()
		except AttributeError:
			pass

		DailyPredictions = pd.DataFrame(columns=['Labels','LastTrainingsDate','DayOfPrediction','PredictedDay','StockPrizeAtDayOfPrediction','PredictedCategory','PredictedProbabilities'],dtype=object)

		for stocklabel in ListOfCompanies:
			
			try:
				tmp = pickle.load(open(self.PathData + 'models/stocks/'+stocklabel+'_model.p'))

			except IOError:
				print "Model for stock",stocklabel, "not found"
				continue


			try: 
				InputData = pd.read_pickle(self.PathData + 'chart/stocks/'+stocklabel+'.p')
			except IOError:
				print "Input data in PredictStocks() for stock ",stocklabel,"is missing"
				continue

			if tmp['LastTrainingsDate'] >= DayOfPrediction:
				print "DayOfPrediction for",stocklabel,"within trainings period, use later date"
				continue 
			
			model = tmp['model']
			ListOfFeatures = tmp['ListOfFeatures']

			InputData = InputData.loc[InputData['Date'] == DayOfPrediction]
			model_input = InputData.loc[:,(InputData.columns.isin(ListOfFeatures) == True) & (InputData.columns.isin(['Date']) == False)].values
			
			model_prediction = model.predict_proba(model_input)[0]
			DailyPredictions = DailyPredictions.append({'Labels':stocklabel,'LastTrainingsDate':tmp['LastTrainingsDate'],
				'DayOfPrediction':DayOfPrediction,
				'PredictedDay':(DayOfPrediction+BDay(10)).date(),
				'StockPrizeAtDayOfPrediction':InputData['Close'].values[0],
				'PredictedCategory':np.argmax(model_prediction),
				'PredictedProbabilities':model_prediction},ignore_index=True)
		
		return DailyPredictions

	def ComputeStockModelsAll(self,ModelType='RFC'):

		self.ComputeStockModels(ListOfTickers=self.ListOfCompanies['Yahoo Ticker'],ModelType=ModelType)


	def ComputeStockModels(self,ListOfTickers,ModelType='RFC'):
		'''
		Update all stock models for companies provided in ListOfCompanies

		The final model is stored in path ".. data/models/stocks/" as pickle file which includes the
		'model': the model object,
		'LastTrainingData': timestamp of last trainings data used
		'timestamp': date when the model has been created			
	
		Parameters
		----------------

		ListOfTickers : List of strings contains yahoo tickers to compute models

		ModelType : string (default = 'RFC') gives the type of model

		'''

		for stocklabel in ListOfTickers:
	
			 params= self.read_modeling_parameters(stocklabel,ModelType=ModelType)
			 if params is None:
			 	continue

			 else:
					ModelingParamters,Features,StartingDate = params

					#try if input/output for model exist
					try:
						input_data = pd.read_pickle(self.PathData + 'chart/stocks/'+stocklabel+'.p')					
						classification_data = pd.read_pickle(self.PathData+'classification/stocks/'+stocklabel+'.p')


					except IOError:
						print "Input data / Classification Data for stock",stocklabel,"does not exist"
						continue

					RFC_ob,LastTrainingsDate = self.SingleRFC(input_data,classification_data,ModelingParamters,Features,EarliestDate=StartingDate)

					pickle.dump({'model':RFC_ob,'LastTrainingsDate':LastTrainingsDate,'timestamp':datetime.datetime.today().date(),'ModelType':ModelType,'ListOfFeatures':Features},open(self.PathData + 'models/stocks/'+stocklabel+'_model.p','wb'))

					print "Final model for",stocklabel,"written"

	def SingleRFC(self,InputData,ClassificationData,ParameterSet,ListOfFeatures,EarliestDate=None,LatestDate=None,n_estimators=200,n_jobs=2):
		'''
		single random forest classification model for given parameter set

		Parameters
		-------------
		InpuData : pandas DataFrame with input data, e.g. chart indicators, etc.

		ClassificationData : pandas DataFrame with stock classification

		ParameterSet : dictionary contains hyperparameters for RFC

		EarliestDate : datetime object (default = None) determines earlist date after which data is used for model

		n_estimator : int (default = 200), number of final trees in RFC

		Returns
		-------------

		RFC : model object which can used to predict future state

		timestamp : datetime object, date of last trading day of trainings data

		'''
	
		if EarliestDate is None:
			EarliestDate = datetime.datetime(2010,1,1)

		if LatestDate is None:
			LatestDate = datetime.datetime.today().date()

		#Prepare InputData and OutputData
		InputData = InputData.loc[:,InputData.columns.isin(ListOfFeatures+['Date']) == True]
		
		_common_dates = util.find_common_notnull_dates(InputData,ClassificationData)
	
		InputData = InputData.loc[(InputData['Date'].isin(_common_dates)) & (InputData['Date'] > EarliestDate) & (InputData['Date'] <= LatestDate)]
		Input = InputData.loc[:,InputData.columns.isin(['Date']) == False].values
	

		ClassificationData = ClassificationData.loc[(ClassificationData['Date'].isin(_common_dates)) & (InputData['Date'] > EarliestDate) & (InputData['Date'] <= LatestDate)]
		Output = np.argmax(ClassificationData.loc[:,ClassificationData.columns.isin(['Date']) == False].values,axis=1)

		if len(Input) != len(Output):
			raise ValueError('Length of input data does not match lenght of output data')

		#double check that  "NaN" values are left over
		if np.any(Input == np.nan) == True:
			raise ValueError('InputData contains "NaN" entries')	
		if np.any(Output== np.nan) == True:
			raise ValueError('OutputData contains "NaN" entries')	

		RFC = RandomForestClassifier(n_estimators=n_estimators,n_jobs=n_jobs,**ParameterSet)

		RFC.fit(Input,Output)

		return RFC,InputData.tail(1)['Date'].tolist()[0].date()

	def read_modeling_parameters(self,tickerSymbol,ModelType='RFC'):
		'''
		reads from predictions.csv that highest score parameter set

		Parameters
		------------
		tickerSymbol : string gives yahoo ticker symbol for stock

		ModelType : string modeltype to seach in ListOfPredictions

		Returns
		------------
		modelParameters : dictionary contains the classification model parameters

		ListOfFeatures : List of strings with used features

		StartingDate : datetime object gives the date to first use the input data

		Example
		------------

		tickerSymbol = 'BOSS.DE'
		
		s = self.read_modeling_parameters(tickerSymbol)
		>>> s[0]
		>>> {'max_features': 'auto', 'max_depth': 70} #dictionary of hyper parameters for RFC 
		>>> s[1]
		>>> ['Close', 'Volume', 'GD200', 'GD100', 'GD50', 'GD38', 'Lower_BB_20_2', 'Upper_BB_20_2', 'RSI_14', 'ADX', 'MACD', 'MAX20', 'MAX65', 'MAX130', 'MAX260', 'MIN20', 'MIN65', 'MIN130', 'MIN260'] #List of input features
		>>> s[2]
		>>> datetime.date(2010, 1, 1) #Earliest time at which trainings data was used

		'''
		if self.ListOfPredictions is None:
			raise ValueError('No list of predictions provided in '+self.PathData+'predictions/predictions_scan.p')

		tmp =self.ListOfPredictions.loc[(self.ListOfPredictions['Labels'] == tickerSymbol) & (self.ListOfPredictions['ModelType'] == ModelType)][['Score','BestParameters','BestParameterValues','StartingDate','ListOfFeatures']]
		
		if len(tmp) == 0:
			print 'No matching ticker found for', tickerSymbol, 'with given modeltype: '+ModelType
			return None

		#find entry with highest score
		tmp= tmp.loc[tmp['Score'].idxmax(),['BestParameters','BestParameterValues','StartingDate','ListOfFeatures']]

		
		BestParameters,BestParameterValues,StartingDate,ListOfFeatures = tmp.values

		modelParamters = {}
		for k in range(len(BestParameters)):
			modelParameters =  modelParamters.update({BestParameters[k]:BestParameterValues[k]})

		return [modelParamters,ListOfFeatures,StartingDate]



class ScanModel(object):

	def __init__(self,ModelType,FileNameListOfCompanies=None,ListOfFeatures='default',GridParameters=None,test_size =0.1,cv_splits=10,n_jobs=-3,PathData=None):
		'''
		The main task of this class is to scan hyper parameters of stocks and save results in file prediction_scan.p

		Parameters
		------------

		ModelType : string indicates type of model, currently implemented is "RFC" for randomforst classification, To do in future: SVM, NeuralNetwork

		FileNameListComapnies : string (defautl = None) file namen for lists stored in data/company_lists/, if None use full list

		ListOfFeatures : List of strings (default ='default'), list of features used for final model, when "default" use all features found in data

		GridParameters : dictionary (default None), dictionary including the hyper parameters for model 

		test_size : float (default = 0.1), fraction of trainings data use for validation

		cv_splits : int (default = 10), number of independent runs for cross validation

		n_jobs : int ( default = -3), number of cores 

		PathData : string (default = None), Path for the folder /data/ 

		'''
		
		self.n_jobs =n_jobs
		self.test_size = test_size
		self.cv_splits = cv_splits
		self.ModelType = ModelType
		self.ListOfFeatures = ListOfFeatures

		self.EarliestDate = datetime.date(2010,1,1)

		if self.ModelType is None or self.ModelType in ['RFC','SVM'] == False:
			raise ValueError('Requested model type is not implemented. Choose "RFC" or "SVM" instead')

		if PathData is None:
			self.PathData = os.path.dirname(os.getcwd())+'/data/'

		else:
			self.PathData = PathData
		
		if FileNameListOfCompanies is None:
			self.FileNameListOfCompanies ='full_list.csv'
		
		else:
			if os.path.isfile(self.PathData+'company_lists/'+FileNameListOfCompanies) is True:
				self.FileNameListOfCompanies = FileNameListOfCompanies
			else:
				raise ValueError('List: '+FileNameListOfCompanies + ' does not exists in' +self.PathData + 'company_lists/')

		
		self.ListOfCompanies = pd.read_csv(self.PathData+'company_lists/'+self.FileNameListOfCompanies,index_col='Unnamed: 0')
		
		if self.ModelType == 'RFC':
			if GridParameters is None:		
				self.ParamGrid = [{'max_depth':[10,20,50,75,100,150,200,300],'max_features':['auto','log2']}]
			else:
				self.ParamGrid = GridParameters

		elif self.ModelType == 'SVM':
			if GridParameters is None:
				self.ParamGrid = [] #To Do

		if os.path.exists(self.PathData + 'predictions') is False:
			os.makedirs(self.PathData + 'predictions')


#		self.val_sets = ShuffleSplit(n_splits = )
	def gridSearchRFC(self,Input,Output,split_sets):
		RFC = RandomForestClassifier(n_estimators=100)
		Grids =GridSearchCV(RFC,self.ParamGrid,cv=split_sets,n_jobs=self.n_jobs)
		Grids.fit(Input,Output)
		return Grids.best_score_,Grids.best_params_

	# To do singel RFC, single SVM, gridSearch SVM
	def StockGridModelingAll(self,scaled=True,EarliestDate=None):
		if EarliestDate is None:
			EarliestDate = self.EarliestDate
		try:
			EarliestDate = EarliestDate.date()
		except AttributeError:
			pass

		self.StockGridModeling(ListOfTickers=self.ListOfCompanies['Yahoo Ticker'],scaled=True,EarliestDate=EarliestDate)

	def StockGridModeling(self,ListOfTickers,scaled=True,EarliestDate=None):

		'''
		Scan hyper parameters for all stocks provided in ListOfCompanyTickers and save results in data/predictions/prediction_scan.p

		Parameters
		-------------

		ListOfCompanyTickers : List of strings, contains stock yahoo tickers

		scaled : bool (default = True) whether or not the input features should be scaled, i.e., subtract mean and divide by standard deviation
		
		EarliestDate : datetime object (default = None), gives earliest date at which data is used as trainings data

		Returns
		--------------
		None

		'''

		if os.path.isfile(self.PathData+'predictions/predictions_scan.p') == False:	
		 	prediction_out = pd.DataFrame(columns=['Labels','ModelType','SearchedParameters','BestParameters','BestParameterValues','Score','Input','ListOfFeatures','Date','StartingDate'],dtype=object)
			prediction_out.to_pickle(self.PathData+'predictions/predictions_scan.p')

		prediction_out = pd.read_pickle(self.PathData+'predictions/predictions_scan.p')

		if EarliestDate is None:
			EarliestDate = self.EarliestDate

		try:
			EarliestDate = EarliestDate.date()
		except AttributeError:
			pass

		for stocklabel in ListOfTickers:
								
			if (os.path.isfile(self.PathData+'chart/stocks/'+stocklabel+'.p') == False) or (os.path.isfile(self.PathData +'classification/stocks/'+stocklabel+'.p') == False):
				print "Data for stock: ",stocklabel, " does not exist"
				continue

			else:

				ChartData = pd.read_pickle(self.PathData+'chart/stocks/'+stocklabel+'.p') 							
				ClassificationData= pd.read_pickle(self.PathData +'classification/stocks/'+stocklabel+'.p') 

				common_dates =util.find_common_notnull_dates(ChartData,ClassificationData)

				if len(common_dates) < 100:
					print "stock",stocklabel,"does not provide enough data for modeling"
					continue
				#get rid of all non common rows in both data sets
				ChartData = ChartData.loc[(ChartData['Date'].isin(common_dates)) & (ChartData['Date']>EarliestDate)]
				ClassificationData = ClassificationData.loc[ClassificationData['Date'].isin(common_dates) & (ClassificationData['Date']>EarliestDate)]

				#check if both files contain same date entries
				if np.any((ChartData['Date'] == ClassificationData['Date']).values == False) == True:
					raise ValueError('Dates of InputData and OutputData do not coincide')

				#double check that  "NaN" values are left over
				if np.any(ChartData.values == np.nan) == True:
					raise ValueError('InputData contains "NaN" entries')	
				if np.any(ClassificationData.values == np.nan) == True:
					raise ValueError('OutputData contains "NaN" entries')	

				'''
				check for feature in featurelist
				'''
				if self.ListOfFeatures is 'default':
					InputFeatures =[_feature for _feature in ChartData.keys() if _feature not in ['Date','Close']]

				else:
					raise ValueError('To do: implemente feature selection')
						
				# set random_state np.random.randint(1,1e6)
				#Xtrain,Xval,Ytrain,Yval = train_test_split(Xfull,Yfull,self.test_size,random_state=1)
				cv = ShuffleSplit(n_splits=self.cv_splits,test_size = self.test_size,random_state = np.random.randint(0,1000000000))

				if self.ModelType == 'RFC':

					#create final numerical input in form of numpy arrays
					Xfull = ChartData.loc[:,ChartData.columns.isin(['Date','Close']) == False].values
				
					Yfull = np.argmax(ClassificationData.loc[:,ClassificationData.columns.isin(['Date']) == False].values,axis=1)

					_out = self.gridSearchRFC(Xfull,Yfull,split_sets=cv)

					#pickle.dump(_out,open('test_11.p','wb'))
					
					#print [self._find_index(prediction_out,n,'S')]
					#search if the same gridsearch has been performed earlier

					mask = util.find_mask(prediction_out,
						[stocklabel,self.ModelType,'Single',self.ParamGrid,InputFeatures,EarliestDate],
						['Labels','ModelType','Input','SearchedParameters','ListOfFeatures','StartingDate'])

					tmp = prediction_out.loc[mask]	

					if len(tmp) == 0:

						prediction_out = prediction_out.append({'Labels':stocklabel,
							'ModelType':self.ModelType,
							'SearchedParameters':self.ParamGrid,
							'BestParameters':_out[1].keys(),
							'BestParameterValues':_out[1].values(),'Score':_out[0],
							'Input':'Single','ListOfFeatures':InputFeatures,
							'Date':datetime.datetime.today().date(),
							'StartingDate':EarliestDate},ignore_index=True)
					
					else:
						
						prediction_out.at[tmp.index.tolist()[0],'BestParameters'] = _out[1].keys()
						prediction_out.at[tmp.index.tolist()[0],'BestParameterValues'] = _out[1].values()
						prediction_out.at[tmp.index.tolist()[0],'Score'] = _out[0]
						prediction_out.at[tmp.index.tolist()[0],'Date'] = datetime.datetime.today().date()
					
				print "Label: ",stocklabel, "prediction done"
			
			#else:
			#	"either input or Output file for stock ", _label, " in Index ",_StockIndex, " is missing"

				prediction_out.to_pickle(self.PathData+'predictions/predictions_scan.p')
				#print prediction_out



