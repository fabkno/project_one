import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import os


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

		os.chdir(path+_index)

		if os.path.isfile('ListOfCompanies.csv'):
			labels = pd.read_csv('ListOfCompanies.csv')['Label']

			for _label in labels:
				stock_prize = pdr.get_data_yahoo(_label,start,end)
				stock_prize.to_csv(_label+'.csv')
				print "Stock ",_label, " updated"
		else:
			print country, ": Index : ",index, " not found"

