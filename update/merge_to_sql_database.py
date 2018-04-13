import pandas as pd
import numpy as np
import sqlite3 as sql
import os,sys
import util
from logger import Log
from datetime import datetime as dt

class MergeToSQL(Log):

	def __init__(self,duration,PathData=None):

		Log.__init__(self,PathData=PathData)

		if PathData is None:
			self.PathData = os.path.dirname(os.getcwd())+'/data/'
		else:
			self.PathData = PathData

		self.PathPrediction = self.PathData+'/predictions/stocks/'

		self.duration = duration
		


		#check if database exists other wise create empty db
		if os.path.isfile(self.PathPrediction +'predictions.db') == False:		
			db = sql.connect(self.PathPrediction+'predictions.db')
			db.close()


		#check if appropriate tables exists otherwise create them
		else:
			db = sql.connect(self.PathPrediction+'predictions.db')

			cursor = db.cursor()
			cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

			tableNames = cursor.fetchall()

			if np.all(np.array([x[0] == "prediction"+str(self.duration)+"BT" for x in tableNames]) == False):

				cursor.execute('''CREATE TABLE prediction'''+str(self.duration)+'''BT(id INTEGER PRIMARY KEY,
				Label TEXT,
				LastTrainingsDate DOB,
				PredictionDay DOB,
				ValidationDay DOB,
				PrizeAtPrediction FLOAT,
				PrizeAtValidation FLOAT,
				RelativePrizeChange FLOAT,
				TrueCategory INT,
				PredictedCategory INT,
				PredictedUpperPrize FLOAT,
				PredictedLowerPrize FLOAT,
				ProbCat0 FLOAT,
				ProbCat1 FLOAT,
				ProbCat2 FLOAT,
				ProbCat3 FLOAT,
				ProbCat4 FLOAT,
				ProbCat5 FLOAT,
				ProbCat6 FlOAT,
				ProbCat7 FLOAT,
				ProbCat8 FLOAT,
				ProbCat9 FLOAT,
				ProbCat10 FLOAT,
				ProbCat11 FLOAT,
				Timestamp DOB)''')
				db.commit()

			db.close()



	def get_tuple_for_db(self,df):
		'''
		get date from pandas dataFrame and return tuple containing data

		values for database include: Label,LastTrainingsDate,PredictionDay,ValidationDay
		                            PrizeAtPrediction,PrizeAtValidation,RelativePrizeChange
		                            TrueCategory, PredictedCategory, Prob Cat 0, 1,2,3,4,5, Timestamp

		Parameters
		-------------
		df : pandas Data sequence

		Returns
		-------------
		tuple
		'''
		
		if len(df.shape) >1:
			raise ValueError('pandas data frame cannot have multiple rows, it must be a pandas sequence')
		if len(df.values[9]) != 12:
			self.logging('Stock '+df.values[0]+': number of predicted probabilities is incorrect !!!')
			print('Stock '+df.values[0]+': number of predicted probabilities is incorrect !!!')
		return tuple(df.values[0:9]) \
				+tuple(df.values[10:12])\
				+tuple([np.round(df.values[9][i],decimals=3) for i in range(12)])+tuple(df.values[14:15])



	def check_if_entry_exists(self,db,data,duration):
		#check if entry exists in database

		cursor = db.cursor()
		Label = data['Labels']
		PredictionDay = data['PredictionDay']   
		

		x1 = db.cursor().execute('''SELECT label,predictionday,prizeatvalidation FROM prediction'''+str(duration)+'''BT WHERE label=? AND predictionday=?''',(Label,PredictionDay))
		
		tmpx1 = x1.fetchall()
		
		#if entry does not exist return False
		if len(tmpx1) == 0:				
			return False

		else:
			#if tmpx1[0][2] is None:
			#	return "update"

			#else:
			return True
			
#		else:
#
#			x2 = db.cursor().execute('''SELECT label,predictionday,validationday,prizeatvalidation FROM prediction'''+str(duration)+'''BT WHERE label=? AND validationday=?''',(Label,PredictionDay))
#			tmpx2= x2.fetchall()
#			
#			if tmpx2[0][3] is None:
#				return "update"
#			
#			else: 
#				return True
			


	def add_PrizeAtValidation_is_nan(self,db,data,duration):
		'''
		check if db contains empty prizeatvalidation, relativeprizechange and truecategory for given predictionday in data

		for example: db contains for entry predictionday = 2017-01-12 no values for validationday = 2017-01-26
		However, the pandas data frame (data) contains data about the predictionday = 2017-01-26 than use these information to update db for predictionday = 2017-01-12
		'''

		Label = data['Labels']

		ValidationDay = data['PredictionDay']
		PrizeAtValidation = data['PrizeAtPrediction']

		x = db.cursor().execute('''SELECT prizeatprediction,prizeatvalidation FROM prediction'''+str(duration)+'''BT WHERE label=? AND validationday =?''',(Label,ValidationDay))

		values = x.fetchone()

		if values is None:
			return

		elif values[1] is None:

			db_PrizeAtPrediction = values[0]
			db_PrizeAtValidation= PrizeAtValidation
			db_RelativePrizeChange = np.round((db_PrizeAtValidation - db_PrizeAtPrediction)/db_PrizeAtPrediction *100.,decimals=3)
			db_TrueCategory = util.find_category(db_RelativePrizeChange)

			db.cursor().execute('''UPDATE prediction'''+str(duration)+'''BT SET prizeatvalidation=? WHERE label=? and validationday=?''',(db_PrizeAtValidation, Label,ValidationDay))
			db.cursor().execute('''UPDATE prediction'''+str(duration)+'''BT SET relativeprizechange=? WHERE label=? and validationday=?''',(db_RelativePrizeChange, Label,ValidationDay))
			db.cursor().execute('''UPDATE prediction'''+str(duration)+'''BT SET truecategory=? WHERE label=? and validationday=?''',(db_TrueCategory, Label,ValidationDay))

			db.commit()

		
			return 

	def add_entry(self,db,pandasDB):

		'''
		adds entry to data base
		'''
		datatuple = self.get_tuple_for_db(pandasDB)
		
		db.cursor().execute('''INSERT INTO prediction'''+str(self.duration)+'''BT(Label,
						LastTrainingsDate,
						PredictionDay,
						ValidationDay,
						PrizeAtPrediction,
						PrizeAtValidation,
						RelativePrizeChange,
						TrueCategory,
						PredictedCategory,
						PredictedUpperPrize,
						PredictedLowerPrize,
						ProbCat0,
						ProbCat1,
						ProbCat2,
						ProbCat3,
						ProbCat4,
						ProbCat5,
						ProbCat6,
						ProbCat7,
						ProbCat8,
						ProbCat9,
						ProbCat10,
						ProbCat11,
						Timestamp
						)
						VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', datatuple)

		db.commit()

	def compare_databases(self,duration,ListOfLabels='all',ListOfPredictionDays='all'):

		self.pandasDBFileName = self.PathPrediction + 'full_predictions_'+str(duration)+'BT.p'

	
		if os.path.isfile(self.pandasDBFileName) == False:
			raise IOError('File in :'+self.pandasDBFileName+' does not exist')		


		df = pd.read_pickle(self.pandasDBFileName)

		if ListOfLabels is 'all':
			ListOfLabels = list(set(df['Labels'].tolist()))
		
		#establish connection to db
		db = sql.connect(self.PathPrediction+'predictions.db')
		

		for label in ListOfLabels:
			print label
			dff = df.loc[df['Labels'] == label]
#			print dff
			if ListOfPredictionDays is not 'all':
				dff = dff.loc[dff['PredictionDay'].isin(ListOfPredictionDays)]
				
			for n in dff.index.tolist():

				if self.check_if_entry_exists(db,dff.loc[n],duration) is False:
					self.add_entry(db,dff.loc[n])
				
				self.add_PrizeAtValidation_is_nan(db,dff.loc[n],duration)

		db.close()

