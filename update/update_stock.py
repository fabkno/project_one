import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import os
from pandas_datareader._utils import RemoteDataError

'''

To do write updater to only update data since execution

'''


ListOfCountries = ['Germany']


start = datetime.datetime(2008,01,01)
end = datetime.datetime.today()

current_path = os.getcwd()

for _country in ListOfCountries:

	os.chdir(current_path)

	path = '../data/raw/'+_country+'/'
	
	ListOfIndices = next(os.walk(path))[1]

	for _index in ListOfIndices:
		os.chdir(current_path)
		os.chdir(path+_index)
	

		if os.path.isfile('ListOfCompanies.csv'):
			labels = pd.read_csv('ListOfCompanies.csv')['Label']

			if len(labels) >=1:

				for _label in labels:
					
					try:
						stock_prize = pdr.get_data_yahoo(_label,start,end)
						stock_prize.to_csv(_label+'.csv')
						print "Stock ",_label, " updated"

					except RemoteDataError:
						print "No information for ticker ", _label
						continue

			else:
				print "Searched index does not have any entries"
		else:
				print country, ": Index : ",index, " not found"

		print "########## Index ", _index, " successfully updated #########\n\n"


