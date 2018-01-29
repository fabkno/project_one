import numpy as np
import pandas as pd
import os
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV
import datetime,sys

class ScanModel(object):

	def __init__(self,ModelType,FileNameListOfCompanies=None,ListOfFeatures='default',GridParameters=None,test_size =0.1,cv_splits=10,n_jobs=-3,PathData=None):

		

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
				self.ParamGrid = [{'max_depth':[10,20,50,70,100,150,200],'max_features':['auto','log2']}]
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


	def StockGridModeling(self,scaled=True):

		if os.path.isfile(self.PathData+'predictions/predictions_scan.p') == False:	
		 	prediction_out = pd.DataFrame(columns=['Labels','ModelType','SearchedParameters','BestParameters','BestParameterValues','Score','Input','ListOfFeatures','Date','StartingDate'],dtype=object)
			prediction_out.to_pickle(self.PathData+'predictions/predictions_scan.p')

		prediction_out = pd.read_pickle(self.PathData+'predictions/predictions_scan.p')

		
		for stocklabel in self.ListOfCompanies['Yahoo Ticker']:
								
			if (os.path.isfile(self.PathData+'chart/stocks/'+stocklabel+'.p') == False) or (os.path.isfile(self.PathData +'classification/stocks/'+stocklabel+'.p') == False):
				print "Data for stock: ",stocklabel, " does not exist"
				continue

			else:
				ChartData = pd.read_pickle(self.PathData+'chart/stocks/'+stocklabel+'.p') 							
				ClassificationData= pd.read_pickle(self.PathData +'classification/stocks/'+stocklabel+'.p') 

				common_dates =self._find_common_notnull_dates(ChartData,ClassificationData)

				#get rid of all non common rows in both data sets
				ChartData = ChartData.loc[(ChartData['Date'].isin(common_dates)) & (ChartData['Date']>self.EarliestDate)]
				ClassificationData = ClassificationData.loc[ClassificationData['Date'].isin(common_dates) & (ClassificationData['Date']>self.EarliestDate)]

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
					InputFeatures =[_feature for _feature in ChartData.keys() if _feature != 'Date']
					
				else:
					raise ValueError('To do: implemente feature selection')
				
				#create final numerical input in form of numpy arrays
				Xfull = ChartData.loc[:,ChartData.columns.isin(['Date','Close']) == False].values
				
				if scaled == True:
					Xfull -= np.mean(Xfull,axis=0)
					Xfull /= np.std(Xfull,axis=0)
				
				Yfull = np.argmax(ClassificationData.loc[:,ClassificationData.columns.isin(['Date']) == False].values,axis=1)
			
				# set random_state np.random.randint(1,1e6)
				#Xtrain,Xval,Ytrain,Yval = train_test_split(Xfull,Yfull,self.test_size,random_state=1)
				cv = ShuffleSplit(n_splits=self.cv_splits,test_size = self.test_size,random_state = True)

				if self.ModelType == 'RFC':
					_out = self.gridSearchRFC(Xfull,Yfull,split_sets=cv)

					#pickle.dump(_out,open('test_11.p','wb'))
					
					#print [self._find_index(prediction_out,n,'S')]
					#search if the same gridsearch has been performed earlier

					mask = self._find_mask(prediction_out,
						[stocklabel,self.ModelType,'Single',self.ParamGrid,InputFeatures,self.EarliestDate],
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
							'StartingDate':self.EarliestDate},ignore_index=True)
						

					
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

	def _find_common_notnull_dates(self,A,B):
		'''
		Returns list common Dates of both dataframes that do not belong to any nan entries

		Parameters
		-----------------
		A : pandas DataFrame

		B : pandas DataFrame

		Returns
		-----------------
		ListOfDates : list of datetimes

		Example
		----------------
		To Do.

		'''
		#find Dates of nonnull entries
		DatesA= A.loc[A.notnull().all(axis=1)]['Date'].tolist()
		DatesB = B.loc[B.notnull().all(axis=1)]['Date'].tolist()

		commonDates = list(set(DatesA) & set(DatesB))
		commonDates.sort()
		return commonDates

	def _find_mask(self,df,ListOfObjects,ListOfColumnNames):

		'''
		To description

		'''

		if len(ListOfObjects) != len(ListOfColumnNames):
			raise ValueError('Caution both number of objects does not match number of ListOfColumnNames')
		
		if len(df) == 0:
			return []

		mask = np.zeros(len(df),dtype=bool) 

		for n in range(len(df)):
			
			row_True = np.zeros(len(ListOfColumnNames))
			
			for k in range(len(ListOfObjects)):
				row_True[k] = df.loc[n][ListOfColumnNames[k]] == ListOfObjects[k]			
			
			mask[n] = np.all(row_True == True)
				

		return mask
