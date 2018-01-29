import numpy as np
import pandas as pd
#util functions 

def find_mask(df,ListOfObjects,ListOfColumnNames):

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

def find_common_notnull_dates(A,B):
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