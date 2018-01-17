import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV


class ScanModel(object):

	def __init__(self,ModelType,ListOfIndices,ListOfFeatures=None,GridParameters=None,test_size =0.1,cv_splits=10,n_jobs=-3,PathData=None):

		self.ListOfIndices = ListOfIndices

		self.n_jobs =n_jobs
		self.test_size = test_size
		self.cv_splits = cv_splits

		self.ModelType = ModelType

		if self.ModelType is None or self.ModelType in ['RFC','SVM'] == False:
			raise ValueError('Requested model type is not implemented. Choose "RFC" or "SVM" instead')

		if ListOfFeatures is None:
			self.ListOfFeatures = "default"

		if PathData is None:
			self.PathData = os.path.dirname(os.getcwd())+'/data/'

		else:
			self.PathData = PathData
		
		if self.ModelType == 'RFC':
			if GridParameters is None:		
				self.ParamGridRFC = [{'max_depth':[10,20,50,70,100,150,200],'max_features':['auto','log2']}]
			else:
				self.ParamGridRFC = GridParameters

		elif self.ModelType == 'SVM':
			if GridParameters is None:
				self.ParamGridSVM = [] #To Do

#		self.val_sets = ShuffleSplit(n_splits = )
	def gridSearchRFC(self,Input,Output,split_sets):
		RFC = RandomForestClassifier(n_estimators=100)
		Grids =GridSearchCV(RFC,self.ParamGridRFC,cv=split_sets,n_jobs=self.n_jobs)
		Grids.fit(Input,Output)
		return Grids.best_score_,Grids.best_params_

	# To do singel RFC, single SVM, gridSearch SVM


	def StockGridModeling(self,scaled=True):

		if os.path.isfile(self.PathData+'predictions/predictions_scan.csv') == False:
		 					prediction_out = pd.DataFrame(columns=['Labels','ModelType','Parameters','ParameterValues','Score','Input','ListOfFeatures'])
							prediction_out.to_csv(self.PathData+'predictions/predictions_scan.csv')

		prediction_out = pd.read_csv(self.PathData+'predictions/predictions_scan.csv',index_col='Unnamed: 0')

		for _country in self.ListOfIndices.keys():

			for _StockIndex in self.ListOfIndices[_country]:

					IndexPath = self.PathData+'raw/'+_country+'/'+_StockIndex+'/'
					IndexPathChart = self.PathData+'chart/'+_country+'/'+_StockIndex+'/'

					if os.path.isfile(IndexPath+'ListOfCompanies.csv'):					
						labels = pd.read_csv(IndexPath+'ListOfCompanies.csv')['Label']


						
						#else:

						#pd.DataFrame(columns=['labels','ModelType'])

						if len(labels) >1:					

							for _label in labels:
								
								
								if (os.path.isfile(IndexPathChart+_label+'_input.csv') == True) and (os.path.isfile(IndexPathChart+_label+'_output.csv') == True):
									
									InputData = pd.read_csv(IndexPathChart+'/'+_label+'_input.csv',index_col='Unnamed: 0') 							
									OutputData= pd.read_csv(IndexPathChart+'/'+_label+'_output.csv',index_col='Unnamed: 0') 


									#check if both files contain same date entries
									if np.any((InputData['Date'] == OutputData['Date']).values == False) == True:
										raise ValueError('Dates of InputData and OutputData to not coincide')

									#double check that  "NaN" values are left over
									if np.any(InputData.values == np.nan) == True:
										raise ValueError('InputData contains "NaN" entries')	

									if np.any(OutputData.values == np.nan) == True:
										raise ValueError('OutputData contains "NaN" entries')	

									'''
									check for feature in featurelist
									'''
									if self.ListOfFeatures == 'default':
										Xfull = InputData.loc[:,InputData.columns.isin(['Date']) == False].values

									else:
										raise ValueError('To do: implemente feature selection')

									if scaled == True:
										Xfull -= np.mean(Xfull,axis=0)
										Xfull /= np.std(Xfull,axis=0)

									Yfull = np.argmax(OutputData.loc[:,OutputData.columns.isin(['Date']) == False].values,axis=1)

									# set random_state np.random.randint(1,1e6)
									#Xtrain,Xval,Ytrain,Yval = train_test_split(Xfull,Yfull,self.test_size,random_state=1)
									cv = ShuffleSplit(n_splits=self.cv_splits,test_size = self.test_size,random_state = True)

									if self.ModelType == 'RFC':

										_out = self.gridSearchRFC(Xfull,Yfull,split_sets=cv)

										tmp= prediction_out.loc[(prediction_out['Labels'] == _label) &
										(prediction_out['ModelType'] == self.ModelType) &
										(prediction_out['Input'] == 'Single') &
										(prediction_out['ListOfFeatures'] == self.ListOfFeatures)
										]

										if len(tmp) == 0:
											prediction_out = prediction_out.append({'Labels':_label,
												'ModelType':self.ModelType,'Parameters':str(_out[1].keys()),
												'ParameterValues':str(_out[1].values()),'Score':_out[0],
												'Input':'Single','ListOfFeatures':self.ListOfFeatures},ignore_index=True)
										else:
											
											prediction_out.loc[tmp.index.tolist(),['Parameters','ParameterValues','Score']] = [str(_out[1].keys()),str(_out[1].values()),_out[0]]

									print "Label: ",_label, "prediction done"
								
								else:
									"either input or Output file for stock ", _label, " in Index ",_StockIndex, " is missing"

		prediction_out.to_csv(self.PathData+'predictions/predictions_scan.csv')
		print prediction_out