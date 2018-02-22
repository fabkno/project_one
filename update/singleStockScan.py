import numpy as np
import pandas as pd
import os
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV
import datetime,sys
import util
from pandas.tseries.offsets import BDay
from logger import Log

class ModelPrediction(Log):
	'''
	Add description ...
	'''
	def __init__(self,FileNameListOfCompanies=None,PathData=None,n_jobs=-2):

		Log.__init__(self,PathData=PathData)

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
				self.logging("ValueError: List :"+FileNameListOfCompanies + "does not exists in " +self.PathData + "company_lists/")
				raise ValueError('List: '+FileNameListOfCompanies + ' does not exists in' +self.PathData + 'company_lists/')

		self.ListOfCompanies = pd.read_csv(self.PathData+'company_lists/'+self.FileNameListOfCompanies,index_col='Unnamed: 0')
		self.n_jobs = n_jobs
		if os.path.exists(self.PathData + 'predictions/stocks/') is False:
			os.makedirs(self.PathData + 'predictions/stocks/')

		if os.path.isfile(self.PathData+'predictions/predictions_scan.p') is True:
			self.ListOfPredictions = pd.read_pickle(self.PathData+'predictions/predictions_scan.p')

		else:
			self.ListOfPredictions = None

		if os.path.exists(self.PathData + 'models/stocks/') is False:
			os.makedirs(self.PathData + 'models/stocks/')

		if os.path.exists(self.PathData + 'simulations/stocks/') is False:
			os.makedirs(self.PathData + 'simulations/stocks/')

		self.UpperCategoryBoundaries = np.array([-5,-2.5,0,2.5,5,100])*0.01
		self.LowerCategoryBoundaries = np.array([-50,-5,-2.5,0,2.5,5])*0.01

	def PredictStocksAll(self,DayOfPrediction=None):

		if DayOfPrediction is None:
			DayOfPrediction = datetime.datetime.today().date()

		predictions = self.PredictStocks(ListOfTickers=self.ListOfCompanies['Yahoo Ticker'],DayOfPrediction = DayOfPrediction)

		predictions.to_pickle(self.PathData+'predictions/stocks/daily_predictions_'+str(DayOfPrediction.year)+'_'+str(DayOfPrediction.month)+'_'+str(DayOfPrediction.day)+'.p')

		print "Daily stock prediction updated"

	def PredictStocks(self,ListOfTickers,DayOfPrediction):
		'''
		write description

		'''

		try: 
			DayOfPrediction = DayOfPrediction.date()
		except AttributeError:
			pass
		DailyPredictions = pd.DataFrame(columns=['Labels','LastTrainingsDate','PredictionDay','ValidationDay',
			'PrizeAtPrediction','PrizeAtValidation','RelativePrizeChange(%)',
			'TrueCategory','PredictedCategory','PredictedProbabilities','PredictedUpperPrice','PredictedLowerPrice','ModelType','ModelParameters'],dtype=object)
		#DailyPredictions = pd.DataFrame(columns=['Labels','LastTrainingsDate','DayOfPrediction','PredictedDay','StockPrizeAtDayOfPrediction','PredictedCategory','PredictedProbabilities','ModelType'],dtype=object)

		for stocklabel in ListOfTickers:
			
			try:
				tmp = pickle.load(open(self.PathData + 'models/stocks/'+stocklabel+'_model.p'))

			except IOError:
				self.logging("IOError: Stock "+stocklabel+": model not found")
				print "Model for stock",stocklabel, "not found"
				continue


			try: 
				input_data = pd.read_pickle(self.PathData + 'chart/stocks/'+stocklabel+'.p')
			except IOError:
				self.logging("IOError: Stock "+stocklabel+": input data is missings")
				print "Input data in PredictStocks() for stock ",stocklabel,"is missing"
				continue

			if tmp['LastTrainingsDate'] >= DayOfPrediction:
				self.logging("Stock "+stocklabel+": DayOfPrediction within trainings period, use later date")
				print "DayOfPrediction for",stocklabel,"within trainings period, use later date"
				continue 
			
			model = tmp['model']
			ListOfFeatures = tmp['ListOfFeatures']		
			modeltype = tmp['ModelType']

			input_data = input_data.loc[input_data['Date'] <=DayOfPrediction].tail(10)
			#input_data = input_data.loc[input_data['Date'] == DayOfPrediction]
		
			if len(input_data.loc[input_data['Date'] == DayOfPrediction]) == 0:
				self.logging('Stock '+stocklabel+': input data for date'+str(DayOfPrediction)+' does not exist yet, take last business day to predict')
				DayOfPrediction = input_data.tail(1)['Date'].tolist()[0].date()
				#DayOfPrediction =(DayOfPrediction - BDay(1)).date()

			print DayOfPrediction
			input_data = input_data.loc[input_data['Date'] == DayOfPrediction]
			
			model_input = input_data.loc[:,(input_data.columns.isin(ListOfFeatures) == True) & (input_data.columns.isin(['Date']) == False)].values
			
			if modeltype != 'RFC':			
				try:
					model_input -= tmp['scaling']['mean']
					model_input /= tmp['scaling']['std']

				except KeyError:
					self.logging("KeyError: Stock "+stocklabel+": no scaling object found, continue without scaling data")
					pass
			
		
			## To do check business days for stock exchange
			model_prediction = model.predict_proba(model_input)[0]
			DailyPredictions = DailyPredictions.append({'Labels':stocklabel,'LastTrainingsDate':tmp['LastTrainingsDate'],
				'PredictionDay':DayOfPrediction,
				'ValidationDay':(DayOfPrediction+BDay(10)).date(),
				'PrizeAtPrediction':np.round(input_data['Close'].values[0],decimals=2),			
				'PredictedCategory':np.argmax(model_prediction),
				'PredictedProbabilities':np.round(model_prediction,decimals=3),
				'PredictedUpperPrice': np.round(input_data['Close'].values[0]*(1+self.UpperCategoryBoundaries[np.argmax(model_prediction)]),decimals=2),
				'PredictedLowerPrice':np.round(input_data['Close'].values[0]*(1+self.LowerCategoryBoundaries[np.argmax(model_prediction)]),decimals=2),
				'ModelType':modeltype,
				'ModelParameters':tmp['ModelParameters']},ignore_index=True)
			
			self.logging("Stock "+stocklabel+": daily prediction done")
		return DailyPredictions

	def ComputeStockModelsAll(self):

		self.ComputeStockModels(ListOfTickers=self.ListOfCompanies['Yahoo Ticker'])


	def ComputeStockModels(self,ListOfTickers,LatestTrainingsDate=None):
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
	
			 params= self.read_modeling_parameters(stocklabel)
			 if params is None:
			 	continue

			 else:
					ModelType,ModelingParameters,Features,StartingDate = params

					#try if input/output for model exist
					try:
						input_data = pd.read_pickle(self.PathData + 'chart/stocks/'+stocklabel+'.p')					
						classification_data = pd.read_pickle(self.PathData+'classification/stocks/'+stocklabel+'.p')


					except IOError:
						self.logging("IOError: Stock "+stocklabel+": input data / classification data not found")
						print "Input data / Classification Data for stock",stocklabel,"does not exist"
						continue

					if ModelType == 'RFC':
						RFC_ob,LastTrainingsDate = self.SingleRFC(input_data,classification_data,ModelingParameters,Features,EarliestDate=StartingDate,LatestTrainingsDate=LatestTrainingsDate)
						pickle.dump({'model':RFC_ob,'LastTrainingsDate':LastTrainingsDate,
							'timestamp':datetime.datetime.today().date(),'ModelType':ModelType,
							'ListOfFeatures':Features,'ModelParameters':ModelingParameters},open(self.PathData + 'models/stocks/'+stocklabel+'_model.p','wb'))

					elif ModelType == 'SVM':
						_out = self.SingleSVM(input_data,classification_data,ModelingParameters,Features,EarliestDate=StartingDate,LatestTrainingsDate=LatestTrainingsDate)
						if len(_out) == 2:
							SVM_ob,LastTrainingsDate= _out
							pickle.dump({'model':SVM_ob,'LastTrainingsDate':LastTrainingsDate,'timestamp':datetime.datetime.today().date(),
								'ModelType':ModelType,'ListOfFeatures':Features,'ModelParameters':ModelingParameters}
								,open(self.PathData + 'models/stocks/'+stocklabel+'_model.p','wb'))

						elif len(_out) == 3:
							SVM_ob,LastTrainingsDate,scaling= _out
							pickle.dump({'model':SVM_ob,'LastTrainingsDate':LastTrainingsDate,'timestamp':datetime.datetime.today().date(),'ModelType':ModelType,
								'ListOfFeatures':Features,'ModelParameters':ModelingParameters,'scaling':scaling},
								open(self.PathData + 'models/stocks/'+stocklabel+'_model.p','wb'))
					
					self.logging("Stock "+stocklabel+": final model written")
					print "Final model for",stocklabel,"written"

	def RunStockSimulation(self,Ticker,StartingDateSimulation,FinalDateSimulation,ModelType=None,scaled=True):


		'''
		runs predictions for old data to compare with real results

		Parameters
		---------------

		Ticker : string, Yahoo stock ticker

		StartingDateSimulation : datetime object, starting date for historic simulations

		FinalDateSimulation : datetime object, final date for historic simulation, when date exceeds historic data 

		'''
		scaling = None

		Validations = pd.DataFrame(columns=['LastTrainingsDate','PredictionDay','ValidationDay','PrizeAtPrediction','PrizeAtValidation',
			'RelativePrizeChange(%)','TrueCategory','PredictedCategory','PredictedProbabilities','PredictedUpperPrice','PredictedLowerPrice',
			'ModelType','ModelParameters'],dtype=object)

		params= self.read_modeling_parameters(Ticker,ModelType=ModelType)
		if params is None:
			self.logging("ValueError: Stock "+Ticker+": parameter values not found in prediction data base")
			raise ValueError("Parameter values for stock",Ticker,"not found in prediction data base")
			
		else:

			ModelType,ModelingParameters,ListOfFeatures,StartingDateTraining = params
				#try if input/output for model exist
		try:
			input_data = pd.read_pickle(self.PathData + 'chart/stocks/'+Ticker+'.p')					
			classification_data = pd.read_pickle(self.PathData+'classification/stocks/'+Ticker+'.p')


		except IOError:
				self.logging("IOError: Stock "+stocklabel+": input data / classification data not found")
				raise ValueError("Input data / Classification Data for stock",Ticker,"does not exist")
		
		#for Star		
		_starting_dates_simulations =  classification_data.loc[(classification_data['Date']>=StartingDateSimulation) &(classification_data['Date']<=FinalDateSimulation)]['Date'].tolist()
		
		for _starting_date in _starting_dates_simulations:
		#print len(input_data.loc[(input_data['Date']>=StartingDateSimulation) &(input_data['Date']<=FinalDateSimulation)]),np.busday_count(FinalDateSimulation,StartingDateSimulation)


			inputSimulation= input_data.loc[(input_data['Date']>=StartingDateTraining) & (input_data['Date']<_starting_date)]
			classificationSimulation =classification_data.loc[(classification_data['Date']>=StartingDateTraining) & (classification_data['Date']<_starting_date)]

			effectiveStartingDate= inputSimulation.tail(1)['Date'].tolist()[0].date()

		 	if ModelType == 'RFC':
				model_ob,LastTrainingsDate = self.SingleRFC(inputSimulation,classificationSimulation,ModelingParameters,ListOfFeatures,EarliestDate=StartingDateTraining,n_estimators=100)

			elif ModelType == "SVM" and scaled == True:
				model_ob,LastTrainingsDate,scaling = self.SingleSVM(inputSimulation,classificationSimulation,ModelingParameters,ListOfFeatures,EarliestDate=StartingDateTraining,scaled=scaled)

			index_tmp = input_data.loc[input_data['Date'] == effectiveStartingDate].index.tolist()[0]
			DayOfPrediction = input_data.loc[index_tmp + 1]['Date'].date()


			model_input = input_data.loc[index_tmp+1,(input_data.columns.isin(ListOfFeatures) == True) & (input_data.columns.isin(['Date']) == False)].values


			if scaling is not None:
				model_input -= scaling['mean']
				model_input /= scaling['std']


			prediction= model_ob.predict_proba(model_input.reshape(1,-1))


			#print classification
			close_at_predictionday = input_data.loc[index_tmp+1]['Close']
			close_at_validationday = input_data.loc[index_tmp+11]['Close']

			Validations = Validations.append({'PredictionDay':DayOfPrediction,
				'LastTrainingsDate':effectiveStartingDate,
				'ValidationDay':input_data.loc[index_tmp+11]['Date'].date(),
				'PrizeAtPrediction':np.round(close_at_predictionday,decimals=2),
				'PrizeAtValidation':np.round(close_at_validationday,decimals=2),
				'RelativePrizeChange(%)':np.round((close_at_validationday-close_at_predictionday)/close_at_predictionday * 100,decimals=2),
				'TrueCategory':np.argmax(classification_data.loc[index_tmp+1,classification_data.columns.isin(['Date']) == False].values),
				'PredictedCategory':np.argmax(prediction),
				'PredictedProbabilities':prediction[0],
				'PredictedUpperPrice': np.round(close_at_predictionday*(1+self.UpperCategoryBoundaries[np.argmax(prediction)]),decimals=2),
				'PredictedLowerPrice':np.round(close_at_predictionday*(1+self.LowerCategoryBoundaries[np.argmax(prediction)]),decimals=2),
				'ModelType':ModelType,
				'ModelParameters':ModelingParameters
				},ignore_index=True)

		#print Validations
		#return Validations
		Validations.to_pickle(self.PathData + 'simulations/stocks/'+Ticker+'_prediction.p')
		self.logging("Stock: "+Ticker+" simlation finished from "+str(StartingDateSimulation) + " until "+str(_starting_dates_simulations[-1].date())+ " with modeltype: "+ModelType)

		print "Finished stock simulation for stock: "+Ticker+" from "+str(StartingDateSimulation) + " until "+str(_starting_dates_simulations[-1].date())+" with modeltype: "+ModelType

	def SingleSVM(self,InputData,ClassificationData,ParameterSet,ListOfFeatures,EarliestDate=None,LatestTrainingsDate=None,n_jobs=2,scaled = True):
		'''
		single SVM (linear or with kernel)

		To Do
		'''
		if EarliestDate is None:
			EarliestDate = datetime.datetime(2010,1,1)

		if LatestDate is None:
			LatestDate = datetime.datetime.today().date()

		#Prepare InputData and OutputData
		InputData = InputData.loc[:,InputData.columns.isin(ListOfFeatures+['Date']) == True]
		
		_common_dates = util.find_common_notnull_dates(InputData,ClassificationData)
	
		InputData = InputData.loc[(InputData['Date'].isin(_common_dates)) & (InputData['Date'] > EarliestDate) & (InputData['Date'] <= LatestTrainingsDate)]
		Input = InputData.loc[:,InputData.columns.isin(['Date']) == False].values
		
		if scaled == True:
			input_mean = np.mean(Input,axis=0)
			input_std = np.std(Input,axis=0)

			Input -=input_mean
			Input /=input_std

		ClassificationData = ClassificationData.loc[(ClassificationData['Date'].isin(_common_dates)) & (InputData['Date'] > EarliestDate) & (InputData['Date'] <= LatestTrainingsDate)]
		Output = np.argmax(ClassificationData.loc[:,ClassificationData.columns.isin(['Date']) == False].values,axis=1)

		util.check_for_length_and_nan(Input,Output)

		SVM = SVC(probability=True,**ParameterSet)

		SVM.fit(Input,Output)

		if scaled == True:
			return SVM,InputData.tail(1)['Date'].tolist()[0].date(),{'mean':input_mean,'std':input_std}
		else:
			return SVM,InputData.tail(1)['Date'].tolist()[0].date()

	def SingleRFC(self,InputData,ClassificationData,ParameterSet,ListOfFeatures,EarliestDate=None,LatestTrainingsDate=None,n_estimators=200,n_jobs=2):
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

		if LatestTrainingsDate is None:
			LatestTrainingsDate = datetime.datetime.today().date()

		#Prepare InputData and OutputData
		InputData = InputData.loc[:,InputData.columns.isin(ListOfFeatures+['Date']) == True]
		
		_common_dates = util.find_common_notnull_dates(InputData,ClassificationData)
	
		InputData = InputData.loc[(InputData['Date'].isin(_common_dates)) & (InputData['Date'] > EarliestDate) & (InputData['Date'] <= LatestTrainingsDate)]
		Input = InputData.loc[:,InputData.columns.isin(['Date']) == False].values
	

		ClassificationData = ClassificationData.loc[(ClassificationData['Date'].isin(_common_dates)) & (InputData['Date'] > EarliestDate) & (InputData['Date'] <= LatestTrainingsDate)]
		Output = np.argmax(ClassificationData.loc[:,ClassificationData.columns.isin(['Date']) == False].values,axis=1)

		util.check_for_length_and_nan(Input,Output)
		# if len(Input) != len(Output):
		# 	raise ValueError('Length of input data does not match lenght of output data')

		RFC = RandomForestClassifier(n_estimators=n_estimators,n_jobs=n_jobs,**ParameterSet)

		RFC.fit(Input,Output)

		return RFC,InputData.tail(1)['Date'].tolist()[0].date()

	def read_modeling_parameters(self,tickerSymbol,ModelType=None):
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
			self.logging("ValueError: No list of predictions provided in "+self.PathData+"predictions/predictions_scan.p")
			raise ValueError('No list of predictions provided in '+self.PathData+'predictions/predictions_scan.p')

		if ModelType is None:
			tmp =self.ListOfPredictions.loc[(self.ListOfPredictions['Labels'] == tickerSymbol) ][['Score','BestParameters','BestParameterValues','StartingDate','ListOfFeatures','ModelType']]
			if len(tmp) == 0:
				self.logging("Stock "+tickerSymbol+": No matching ticker found")
				print 'No matching ticker found for', tickerSymbol
				return None
		
		else:
			tmp =self.ListOfPredictions.loc[(self.ListOfPredictions['Labels'] == tickerSymbol) & (self.ListOfPredictions['ModelType'] == ModelType)][['Score','BestParameters','BestParameterValues','StartingDate','ListOfFeatures','ModelType']]
			if len(tmp) == 0:
				self.logging("Stock "+tickerSymbol+": No matching ticker found for given modeltype: "+ModelType)
				print 'No matching ticker found for', tickerSymbol, 'with given modeltype: '+ModelType
				return None
	

		#find entry with highest score
		tmp= tmp.loc[tmp['Score'].idxmax(),['BestParameters','BestParameterValues','StartingDate','ListOfFeatures','ModelType']]

		
		BestParameters,BestParameterValues,StartingDate,ListOfFeatures,ModelType = tmp.values

		modelParamters = {}
		for k in range(len(BestParameters)):
			modelParameters =  modelParamters.update({BestParameters[k]:BestParameterValues[k]})

		return [ModelType,modelParamters,ListOfFeatures,StartingDate]



class ScanModel(Log):

	def __init__(self,ModelType,FileNameListOfCompanies=None,ListOfFeatures='default',GridParameters=None,test_size =0.1,cv_splits=5,n_jobs=-2,PathData=None):
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
		
		Log.__init__(self,PathData=PathData)

		self.n_jobs =n_jobs
		self.test_size = test_size
		self.cv_splits = cv_splits
		self.ModelType = ModelType
		self.ListOfFeatures = ListOfFeatures

		self.EarliestDate = datetime.date(2010,1,1)

		if self.ModelType is None or self.ModelType in ['RFC','SVM'] == False:
			self.logging("ValueError: Requested model type is not implemented. Choose >>RFC<< or >>SVM<< instead")
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
				self.logging("ValueError: List "+FileNameListOfCompanies +" does not exist in "+self.PathData +"company_lists/")
				raise ValueError('List: '+FileNameListOfCompanies + ' does not exist in' +self.PathData + 'company_lists/')

		
		self.ListOfCompanies = pd.read_csv(self.PathData+'company_lists/'+self.FileNameListOfCompanies,index_col='Unnamed: 0')
		
		if self.ModelType == 'RFC':
			if GridParameters is None:		
				self.ParamGrid = [{'max_depth':[10,20,50,75,100,150,200,300],'max_features':['auto','log2']}]
			else:
				self.ParamGrid = GridParameters

		elif self.ModelType == 'SVM':
			if GridParameters is None:
				self.ParamGrid = [{'C':[1e-2,1e-1,1,5,1e1,1e2,1e3,5e3,1e4],'gamma':[1e-2,1e-1,0.5,1,5,1e1,50],'kernel':['rbf']}] 
				
			else:
				self.ParamGrid


		if os.path.exists(self.PathData + 'predictions') is False:
			os.makedirs(self.PathData + 'predictions')


#		self.val_sets = ShuffleSplit(n_splits = )
	def gridSearch(self,Input,Output,split_sets,modeltype='RFC'):
		'''
		grid search function for random forest classifier

		'''
		if modeltype == 'RFC':
			ob = RandomForestClassifier(n_estimators=100)
		
		elif modeltype == 'SVM':
			ob = SVC()

		Grids =GridSearchCV(ob,self.ParamGrid,cv=split_sets,n_jobs=self.n_jobs)
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
				self.logging("Stock "+stocklabel+": chart or classification data does not exist")
				print "Data for stock: ",stocklabel, " does not exist"
				continue

			else:

				ChartData = pd.read_pickle(self.PathData+'chart/stocks/'+stocklabel+'.p') 							
				ClassificationData= pd.read_pickle(self.PathData +'classification/stocks/'+stocklabel+'.p') 

				common_dates =util.find_common_notnull_dates(ChartData,ClassificationData)

				if len(common_dates) < 100:
					self.logging("Stock "+stocklabel+": not enough data provided for modeling (<100 trainings days)")
					print "stock",stocklabel,"does not provide enough data for modeling"
					continue
				#get rid of all non common rows in both data sets
				
				ChartData = ChartData.loc[(ChartData['Date'].isin(common_dates)) & (ChartData['Date']>EarliestDate)]
				ClassificationData = ClassificationData.loc[ClassificationData['Date'].isin(common_dates) & (ClassificationData['Date']>EarliestDate)]
				
				#check if both files contain same date entries
				util.check_for_length_and_nan(ChartData,ClassificationData)

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



				#create final numerical input in form of numpy arrays

				Xfull = ChartData.loc[:,ChartData.columns.isin(['Date','Close']) == False].values

				if scaled is True:
					Xfull -= np.mean(Xfull,axis=0)
					Xfull /= np.std(Xfull,axis=0)

				Yfull = np.argmax(ClassificationData.loc[:,ClassificationData.columns.isin(['Date']) == False].values,axis=1)

				_out = self.gridSearch(Xfull,Yfull,split_sets=cv,modeltype=self.ModelType)

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
				
				self.logging("Stock "+stocklabel+": prediction done")
				print "Label: ",stocklabel, "prediction done"
			
			#else:
			#	"either input or Output file for stock ", _label, " in Index ",_StockIndex, " is missing"

				prediction_out.to_pickle(self.PathData+'predictions/predictions_scan.p')
				#print prediction_out



