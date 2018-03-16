import numpy as np
import pandas as pd
import util
from logger import Log
import os
from bs4 import BeautifulSoup
import requests
import datetime
class AnalystsComments(Log):
	'''

	class that manages stock analysis given by professional stock analysts. More precisely, the class includes a web scrapter for finanzen.net and processing method

	'''
	def __init__(self,FileNameListOfCompanies=None,PathData=None):

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



		URL_list = self.ListOfCompanies['URL'].tolist()

		URL_list = [url.replace('aktien','analysen').replace('Aktie','Analysen') for url in URL_list]

		self.ListOfCompanies.loc[:,'URL'] = URL_list

		if os.path.exists(self.PathData + 'analysts_comments/stocks/') is False:
			os.makedirs(self.PathData + 'analysts_comments/stocks/')


	def web_updater(self,ListOfTickers):

		for stocklabel in ListOfTickers:

			print "Start Stock:",stocklabel
			try: 
				db = pd.read_pickle(self.PathData+'analysts_comments/stocks/'+stocklabel+'.p')
				newest_Date,newest_Analyst = db.head(1)[['Date','Analyst']].values[0]

			except IOError:

				db = pd.DataFrame(columns=['Date','TargetPrize','FormerTargetPrize','Category','Analyst','Phrase'])		
				newest_Date = None
				newest_Analyst = None

			try:
				url = self.ListOfCompanies.loc[self.ListOfCompanies['Yahoo Ticker'] == stocklabel]['URL'].values[0]
			
			except IndexError:
				print('URL for stock: '+stocklabel+ ' not found in loaded ListOfCompanies')
				continue

			if '.' in stocklabel:
				currency = 'Euro'
			else:
				currency = ['US-Dollar','USD']

			'''
			start looping through all subwebsite (int_page number on finanzen.net),
			check after every subsite if entries have been checked already
			'''
			#newest_Date = datetime.datetime(2018,3,9).date()
			#newest_Analyst = "DZ BANK"


			k = 1

			while_break = 0

			while( while_break != 1):
				print k, newest_Date,newest_Analyst

				page = requests.get(url+"@intpagenr_"+str(k))

				soup = BeautifulSoup(page.content,'html.parser')

				tmpDB = pd.DataFrame(columns=['Date','TargetPrize','FormerTargetPrize','Category','Analyst','Phrase'])

				for num,l in enumerate(soup.findAll("table")[1].findAll("tr")):
					sub_url =  "https://www.finanzen.net"+l.findAll('a')[0].get('href')

					tds = l.findAll('td')
					
					s = tds[0].get_text().split('.')

					if s[0][-3:] == 'Uhr':
						date = datetime.datetime.today().date()

					else:
						if len(s[2]) == 2:
							s[2] = "20"+s[2]
						date = datetime.datetime(int(s[2]),int(s[1]),int(s[0])).date()


					Category = tds[1].get_text().split(' ')[1]
					Analyst = tds[2].get_text()

					if newest_Date == date and newest_Analyst == Analyst:
						while_break =1
						break

					detailedPage = requests.get(sub_url)
					soupNew = BeautifulSoup(detailedPage.content,'html.parser')

					string= soupNew.findAll("div",{"class":" teaser teaser-xs color-news"})[0].get_text()

					dic = self._get_phrase(string,currency)
					if dic is None:
						continue

				

					tmpDB = tmpDB.append({'Date':date,'Category':Category,'Analyst':Analyst,
					'Phrase':dic['Phrase'],'TargetPrize':dic['TargetPrize'],
					'FormerTargetPrize':dic['FormerTargetPrize']},ignore_index=True)

					
				tmpDB.drop_duplicates(inplace=True)

				if newest_Date is None:
					newest_Date,newest_Analyst = tmpDB.head(1)[['Date','Analyst']].values[0]

				db = pd.concat([db,tmpDB],ignore_index=True)

				k+=1


			db.sort_values(by=['Date'],ascending=False,inplace=True)
			db.drop_duplicates(inplace=True)
			db.reset_index(drop=True,inplace=True)

			db.to_pickle(self.PathData + 'analysts_comments/stocks/'+stocklabel+'.p')
			
	def _check_URLs(self,ListOfURLs=None):

		if ListOfURLs is None:
			ListOfURLs = self.ListOfCompanies['URL'].tolist()

		return util.URL_online(ListOfURLs)

	def _get_phrase(self,string,currency):
		'''
		extracts information from string (text block including the stock analysis)

		Parameters
		--------------
		string 

		Returns
		--------------
		dictionary

    	'''
		string = string.replace(',','.')
		ListOfStrings=  string.split(" ")
		indices = [i for i, x in enumerate(ListOfStrings) if x in currency]
		if len(indices) == 0:
			#print("No currency string found ")
			return None
		
		else:
			for ind in indices:
				try:
					float(ListOfStrings[ind-1])
					break

				except ValueError:
					if ind == indices[-1]:
						return None
					else:
						continue

		target = float(ListOfStrings[ind-1])
		formerTarget = None

		

		if ListOfStrings[ind-3].isdigit() is True:
			phrase = " ".join(ListOfStrings[ind-4:ind+2])
			formerTarget = float(ListOfStrings[ind-3])
		else:
			phrase = " ".join(ListOfStrings[ind-2:ind+2])
			formerTarget = target

		return {"Phrase":phrase,"TargetPrize":target,'FormerTargetPrize':formerTarget}