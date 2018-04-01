import numpy as np
import pandas as pd
import os,sys
from logger import Log
import requests
from bs4 import BeautifulSoup
import util

class ShortsUpdater(Log):
	'''
	This class is designed to update information of short positions acquired by hedge fonds


	'''
	def __init__(self,PathData = None):

		Log.__init__(self,PathData=PathData)

		if PathData is None:
			self.PathData = os.path.dirname(os.getcwd())+'/data/'
		else:
			self.PathData = PathData
	

		self.ListOfCompanies = pd.read_csv(self.PathData + 'company_lists/full_list.csv',index_col='Unnamed: 0')


	def update_from_file(self,filename = None):

		if filename is None:
			filename = 'shorts_list.csv'

		if os.path.isfile(self.PathData+'shorts/'+filename) is False:
			raise ValueError('List of shorts in '+'shorts/'+filename+' does not exist')

		Shorts = pd.read_csv(self.PathData+'shorts/'+filename,decimal=",")

		ListOfShorts = pd.DataFrame(columns=['Date'])
		ListOfShorts['Date'] = pd.date_range(Shorts.tail(1)['Datum'].values[0],Shorts.head(1)['Datum'].values[0])

		# get unique list of isin's
		isin_list = list(set(Shorts['ISIN'].tolist()))


		for isin in isin_list:

			try:
				label= self.ListOfCompanies.loc[self.ListOfCompanies['ISIN'] == isin]['Yahoo Ticker'].values[0]
			except IndexError:
				print "ISIN", isin," for stock: ",Shorts.loc[Shorts['ISIN'] ==isin].head(1)['Emittent'].values[0]," not found in data base"

				continue

			#get unique list of owners for given isin
			owners = list(set(Shorts.loc[Shorts['ISIN'] == isin]['Positionsinhaber'].tolist()))

			print "Label found for ISIN",label

			ListOfShorts[label] = pd.Series(0,index=ListOfShorts.index)

			for owner in owners:
				tmp = Shorts.loc[(Shorts['ISIN'] == isin) &(Shorts['Positionsinhaber'] == owner)]

				for i in range(1,len(tmp)+1):
					if i == len(tmp):
						ListOfShorts.loc[ListOfShorts['Date'] >= tmp['Datum'].tail(i).values[0],label] += tmp.tail(i)['Position'].values[0]  
					else:
						ListOfShorts.loc[(ListOfShorts['Date'] >= tmp['Datum'].tail(i).values[0]) &(ListOfShorts['Date'] < tmp['Datum'].tail(i+1).values[0]),label] += tmp.tail(i)['Position'].values[0]
			
		ListOfShorts.to_pickle(self.PathData+'shorts/list.p')

	def update_from_web(self,filename=None):
		"""
		update latest short positions from bundesanzeiger.de and add to list of shorts

		"""
		if filename is None:
			filename = 'shorts_list.csv'

		if os.path.isfile(self.PathData+'shorts/'+filename) is False:
			raise ValueError('List of shorts in '+'shorts/'+filename+' does not exist. Run "update_from_file()" before')

		Shorts = pd.read_csv(self.PathData+'shorts/'+filename)


		#### web scraper for bundesanzeiger ####
		url = 'https://www.bundesanzeiger.de/'

		if util.URL_online(url) == False:
			self.logging("Error: url "+url+" is currently not online")
			raise ValueError("url "+url+" is currently not online")

		s = requests.Session()
		r = s.get(url)
		soup = BeautifulSoup(r.content, 'html.parser')

		for link in list(soup.find_all('a')):
			if "Leerverkaufspositionen" in str(link):
				r1 =s.get("https://www.bundesanzeiger.de"+link.get('href'))
			
		
		soup =BeautifulSoup(r1.content,'html.parser')
		l = soup.findAll("table", {"class": "result"})[0]
		names = l.findAll("td",{"class":"first"})
		pos = l.findAll("td",{"class":"col_nlp_position"})
		date = l.findAll("td",{"class":"last"})
		comp_data = l.findAll("td",{"class":None})

		if len(names)*2 != len(comp_data):
			self.logging("Caution probable error found, length of found short positions does not equal length of found ISINs")
			print "Caution probable error found, length of found short positions does not equal length of found ISINs"

		#initialize empty data frame
		tmp = pd.DataFrame(columns=['Positionsinhaber','Emittent','ISIN','Position','Datum'],dtype=object)
		
		#append found entries from web site
		for i in range(len(names)):
			tmp = tmp.append({'Positionsinhaber':names[i].get_text(),
								'Position':pos[i].get_text()[0:-2],
								'Datum':date[i].get_text(),
								'ISIN':comp_data[i*2+1].get_text(),
								'Emittent':comp_data[i*2].get_text()[0:util.find_str(comp_data[i*2].get_text(),"Historie",return_ind="start")-2]},ignore_index=True)
	
		N_shorts = len(Shorts)
		Shorts = pd.concat([tmp,Shorts],ignore_index=True)
		Shorts.drop_duplicates(inplace=True)
		Shorts.reset_index(drop=True,inplace=True)

		Shorts.to_csv(self.PathData+'shorts/'+filename,index=False,encoding='utf-8')
		print len(Shorts) - N_shorts, " new short positions found"

		if (len(Shorts) - N_shorts) >=19:
			self.logging("More than 19 new entries found, download new list from bundesanzeiger.de")
			print("more than 19 new entries found, download new list from bundesanzeiger.de")



