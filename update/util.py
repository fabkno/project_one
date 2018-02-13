import numpy as np
import pandas as pd
import requests
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

def check_for_length_and_nan(A,B):
	'''
	checks that both data frames have equal length and do not contain any NaNs

	Parameters
	--------------

	A : pandas DataFrame or nd.array

	B : pandas DataFrame or nd.array

	Returns
	-------------
	None 
	'''

	if len(A) != len(B):
		raise ValueError('Length of input data does not match lenght of output data')

	#double check that  "NaN" values are left over

	if isinstance(A,pd.DataFrame) == True and isinstance(B,pd.DataFrame) == True:

		if pd.isnull(A).values.any() == True:
			raise ValueError('InputData contains "NaN" entries')	
		if pd.isnull(A).values.any() == True:
			raise ValueError('OutputData contains "NaN" entries')	
	elif isinstance(A,np.ndarray) == True and isinstance(B,np.ndarray) == True:

		if np.any(A == np.nan) is True:
			raise ValueError('InputData contains "NaN" entries')
		if np.any(B == np.nan) is True:
			raise ValueError('OutputData contains "NaN" entries')

	else:

		raise ValueError("objects are not of the same type")

def find_str(s, char,return_ind='end'):

	'''
	finds position of substring in string returns position of last found character 

	Parameters
	----------------

	s string : string in which to look for substring

	char string : substring 

	return_ind : string (default = 'end' other option is 'start') returns either end or start index of searched char

	Returns
	----------------
	index of first character of substring is found

	Example
	----------------

	s = "Zur Aktie Dialog SemiconductorWKN927200ISINGB0059822006Deutsches SymbolDLGIndizes TecDAX, Prime All Share, Late TecDAX, Technology All Share, TecDAX Kursindex, BX Swiss"

	substring = "Indizes"

	>>> find_str(s,substring)

	>>> 82
	>>> s[82:] 
	>>> TecDAX, Prime All Share, Late TecDAX, Technology All Share, TecDAX Kursindex, BX Swiss

	'''

	index = 0

	if char in s:
		c = char[0]
		for ch in s:
			if ch == c:
				if s[index:index+len(char)] == char:
					if return_ind == 'end':
						return index+len(char)+1
					elif return_ind =='start':
						return index 

			index += 1

	return -1

def URL_online(URL):
	'''
	check if given URL or list of URLs is online

	Parameters
	-------------

	URL either string or list of strings

	Returns
	-------------

	bool or boolian np.array

	'''
	if isinstance(URL,str):
		page = requests.get(URL)

		if page.status_code != 200:
			return False
		else:
			return True
	else:
		return_values = np.zeros(len(URL),dtype=bool)
		for i in range(len(URL)):

			if requests.get(URL[i]).status_code == 200:
				return_values[i] = True
		return return_values