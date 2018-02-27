import pandas as pd
import numpy as np
import sqlite3 as sql
import os,sys
import util

from datetime import datetime as dt

prediction_path = '../data/predictions/stocks/'

pandasDBFileName = prediction_path +'full_predictions_10BT.p'

if os.path.isfile(pandasDBFileName) == False:
	raise IOError('File in :'+pandasDBFileName+' does not exist')

#check if database exists other wise create empty db
if os.path.isfile(prediction_path +'predictions.db') == False:
	
	db = sql.connect(prediction_path+'predictions.db')

	cursor = db.cursor()

	cursor.execute('''CREATE TABLE prediction10BT(id INTEGER PRIMARY KEY,
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
	Timestamp DOB)''')
	db.commit()

	db.close()




def get_tuple_for_db(df):
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

   return tuple(df.values[0:9]) \
          +tuple(df.values[10:12])\
          +tuple([np.round(df.values[9][i],decimals=3) for i in range(6)])+tuple(df.values[14:15])



def check_if_entry_exists(db,data):
    #check if entry exists in database
    
    cursor = db.cursor()
    Label = data['Labels']
    PredictionDay = data['PredictionDay']   
        
    x = db.cursor().execute('''SELECT label,predictionday FROM prediction10BT WHERE label=? AND predictionday=?''',(Label,PredictionDay))
    
    if len(x.fetchall()) == 0:
		return False
    else: return True


def add_PrizeAtValidation_is_nan(db,data):
	'''
	check if db contains empty prizeatvalidation, relativeprizechange and truecategory for given predictionday in data

	for example: db contains for entry predictionday = 2017-01-12 no values for validationday = 2017-01-26
	However, the pandas data frame (data) contains data about the predictionday = 2017-01-26 than use these information to update db for predictionday = 2017-01-12
	'''

	Label = data['Labels']

	ValidationDay = data['PredictionDay']
	PrizeAtValidation = data['PrizeAtPrediction']

	x = db.cursor().execute('''SELECT prizeatprediction,prizeatvalidation FROM prediction10BT WHERE label=? AND validationday =?''',(Label,ValidationDay))

	values = x.fetchone()

	if values is None:
		return

	elif values[1] is None:

		db_PrizeAtPrediction = values[0]
		db_PrizeAtValidation= PrizeAtValidation
		db_RelativePrizeChange = np.round((db_PrizeAtValidation - db_PrizeAtPrediction)/db_PrizeAtPrediction *100.,decimals=3)
		db_TrueCategory = util.find_category(db_RelativePrizeChange)

		db.cursor().execute('''UPDATE prediction10BT SET prizeatvalidation=? WHERE label=? and validationday=?''',(db_PrizeAtValidation, Label,ValidationDay))
		db.cursor().execute('''UPDATE prediction10BT SET relativeprizechange=? WHERE label=? and validationday=?''',(db_RelativePrizeChange, Label,ValidationDay))
		db.cursor().execute('''UPDATE prediction10BT SET truecategory=? WHERE label=? and validationday=?''',(db_TrueCategory, Label,ValidationDay))

		db.commit()
		return 

def add_entry(db,pandasDB):

	'''
	adds entry to data base
	'''
	datatuple = get_tuple_for_db(pandasDB)
	db.cursor().execute('''INSERT INTO prediction10BT(Label,
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
					Timestamp
					)
					VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', datatuple)

	db.commit()

def compare_databases(ListOfLabels='all',ListOfPredictionDays='all'):

	df = pd.read_pickle(pandasDBFileName)

	if ListOfLabels is 'all':
		ListOfLabels = list(set(df['Labels'].tolist()))
	
	#establish connection to db
	db = sql.connect(prediction_path+'predictions.db')
	

	for label in ListOfLabels:

		dff = df.loc[df['Labels'] == label]

		if ListOfPredictionDays is not 'all':
			dff = dff.loc[dff['PredictionDay'].isin(ListOfPredictionDays)]

		for n in dff.index.tolist():

			if check_if_entry_exists(db,dff.loc[n]) is False:
				add_entry(db,dff.loc[n])

			add_PrizeAtValidation_is_nan(db,dff.loc[n])

	db.close()

