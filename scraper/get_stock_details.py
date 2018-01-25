import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import os
import re



class UpdateStockDetails(object):
	'''

	This classs is designed to update all stock information (usually not on a daily base) execept for the daily stock values

	
	'''
	def __init__(self,Path=None):

		self.today = datetime.datetime.today().date()

		if Path is None:
			self.Path = os.getcwd()
		else:
			self.Path = Path
		
		if os.path.isfile(self.Path + 'ListOfCompanies.p') == True:
			self.ListOfCompanies = pd.read_pickle(self.Path + 'ListOfCompanies.p')

			#add new columns
			for category in ['ISIN','WKN','SymbolFinanzen.net','Country','ListedIndizes','PeerGroupFinanzenNetName','PeerGroupFinanzenNetURLs','PeerGroupFinanzenNetOnline']:

				if category not in self.ListOfCompanies.keys() and category in ['ListedIndizes','PeerGroupFinanzenNetName','PeerGroupFinanzenNetURLs','PeerGroupFinanzenNetOnline']:
					self.ListOfCompanies[category] = pd.Series(np.nan,index=self.ListOfCompanies.index,dtype=object)

				elif category not in self.ListOfCompanies.keys() and category not in ['ListedIndizes','PeerGroupFinanzenNetName','PeerGroupFinanzenNetURLs','PeerGroupFinanzenNetOnline']:
					self.ListOfCompanies[category] = pd.Series(np.nan,index=self.ListOfCompanies.index)

		# To do check time stamp
		elif os.path.isfile(self.Path +'CompaniesByBranches.p') == True:
			self.CompaniesByBranches = pd.read_pickle(self.Path + 'CompaniesByBranches.p')


		

	def add_stock_details(self):

		'''
		To do
		'''

		for N in range(401,1000,20):
			print N,"\n"
			details = self._find_stock_details_from_url(self.ListOfCompanies.loc[N]['URL'])

						
			self.ListOfCompanies.loc[N,['ISIN','WKN','SymbolFinanzen.net','Country']] = [details['ISIN'],details['WKN'],details['SymbolFinanzen.net'],details['Country']]
			self.ListOfCompanies.at[N,'ListedIndizes'] = details['ListedIndizes']
			

			_urls = details['PeerGroupFinanzenNetURLs']
			_names = details['PeerGroupFinanzenNetName']


			if _names is not None:

				_online = np.zeros(len(_names),bool)

				for _n,_url in enumerate(_urls):
					_tmp = self.ListOfCompanies.loc[self.ListOfCompanies['URL'].str.lower() == _url.lower()]
					if len(_tmp) > 0:
						if _names[_n] != _tmp['Name'].values[0]:
							_names[_n] = _tmp['Name'].values[0]

						_online[_n] = True

					else:
						print "company ", _names[_n], " added"
						self.ListOfCompanies = self.ListOfCompanies.append({'Name':_names[_n],'URL':_url,'Branche':None,
											'ISIN':np.nan,'WKN':np.nan,'SymbolFinanzen.net':np.nan,
											'Country':np.nan,'ListedIndizes':np.nan,'PeerGroupFinanzenNetName':np.nan,
											'PeerGroupFinanzenNetURLs':np.nan,'PeerGroupFinanzenNetOnline':np.nan},ignore_index=True)


				_mask = np.where(_online == False)[0]
				_online[_mask] = self._URL_online(np.array(_urls)[_mask])

			else:
				_online=None

			
			

			self.ListOfCompanies.at[N,'PeerGroupFinanzenNetName'] = _names
			self.ListOfCompanies.at[N,'PeerGroupFinanzenNetURLs'] = _urls
			self.ListOfCompanies.at[N,'PeerGroupFinanzenNetOnline'] = _online

			print self.ListOfCompanies.loc[N][['Name','Country','SymbolFinanzen.net','PeerGroupFinanzenNetName','PeerGroupFinanzenNetOnline']]

		#companies.to_pickle('ListofCompanies.p')


	def _find_stock_details_from_url(self,url):

		'''
		function to update stock detail from finanzen.net 
		
		implemented features are: WSK,ISIN,Symbol,Country,ListedIndizes,PeerGroupFinanzenNetName,PeerGroupFinanzenNetURLs
		
		Parameters
		------------

		url : string 

		Returns
		------------

		dict : dictionary with extracted features


		'''

		page = requests.get(url)
		if page.status_code != 200:
			return None
		

		#find WKN,ISIN,SYMBOL if exists
		symbol = None
		soup = BeautifulSoup(page.content, 'html.parser')

		allIdentifiers= list(soup.findAll('title'))
		identifier= allIdentifiers[0].get_text().split()[-1].split(',')

		wkn = identifier[0][1:]

		if len(identifier) == 2:
			isin = identifier[1][:-1]
		elif len(identifier) == 3:
			isin = identifier[2][:-1]
			symbol = identifier[1]

		#find country of company and all stock indices that list stock
		allIndices = list(soup.findAll("div", {"class": "box"}))

		# prepare string to find peer group information
		__tmp = url.rsplit('/', 1)[-1]

		__tmp2 = " ".join(re.split('_|-',__tmp))

		#capitalize first word
		__Listtmp2 = __tmp2.split()
		__Listtmp2[0] = __Listtmp2[0].upper()

		string4PeerGroup = "Peer Group " + __tmp2
		string4PeerGroup2 = "Peer Group " + ' '.join(__Listtmp2)

		string4PeerGroup= string4PeerGroup.encode('latin1')
		string4PeerGroup2= string4PeerGroup2.encode('latin1')


		PeerList = None
		PeerURLList = None
		for name in allIndices:

			if "Zur Aktie" in str(name):#Zum Unternehmen       
				zurAktie = name.get_text()
				num_ = self._find_str(zurAktie,"Indizes")

				if symbol is None:
					num1_ = self._find_str(zurAktie,"Symbol",return_ind='end')
					num2_ = self._find_str(zurAktie,"Indizes",return_ind='start')
					symbol = zurAktie[num1_-1:num2_]
					
					if len(symbol) <3:
						symbol = None

				if num_ != -1:
					ListedIndizes = zurAktie[num_:].split(',')
				else:
					ListedIndizes = None

			elif "Zum Unternehmen" in str(name):
				zumUnternehmen = name.get_text()
				Land = zumUnternehmen[self._find_str(zumUnternehmen,'Land')-1:self._find_str(zumUnternehmen,'Branchen')-9]


			elif string4PeerGroup in str(name) or string4PeerGroup2 in str(name):
				zurPeerGroup = name.get_text()
				PeerList,PeerURLList = self._get_peer_company_names(zurPeerGroup,string4PeerGroup)

		# print out single found features
		if  1 == 0:
			print "WKN " ,wkn
			print "ISIN" , isin
			print 'SymbolFinanzen.net ', symbol
			print 'Country ', Land
			print 'ListedIndizes', ListedIndizes
			print 'PeerGroupFinanzenNetName', PeerList

		return {'WKN':wkn,'ISIN':isin,'SymbolFinanzen.net':symbol,'Country':Land,'ListedIndizes':ListedIndizes,'PeerGroupFinanzenNetName':PeerList,'PeerGroupFinanzenNetURLs':PeerURLList}





	def _get_peer_company_names(self,webstring,start_name):

		'''
		helper function that finds from website string the finanzen.net url
		Parameters
		-------------

		string : raw string extracted from website

		start_name : string "Peer Group" + company name

		Returns
		-------------
		PeerList : list of strings, peer group company names 

		PeerURLList, list of strings, corresponding finanzen.net urls

		
		Example
		-------------

		webstring = "Peer Group Dover AktieDanaher81,4 EUR+0,5%Gardner Denver29,6 EUR+0,0%Illinois Tool Works140,0 EUR-0,7%Zebra Technologies99,0 EUR-2,9%"

		start_name = "Peer Group Dover Aktie"

		PeerNames,PeerURLs = _get_peer_company_names(webstring,start_name)

		>>> PeerNames
		>>> [u'Danaher', u'Gardner Denver', u'Illinois Tool Works', u'Zebra Technologies']

		>>> PeerURLs
		>>> [u'http://www.finanzen.net/aktien/Danaher-Aktie', u'http://www.finanzen.net/aktien/Gardner_Denver-Aktie', u'http://www.finanzen.net/aktien/Illinois_Tool_Works-Aktie', u'http://www.finanzen.net/aktien/Zebra_Technologies-Aktie']

		'''
		PeerList = []
		PeerURLList = []
		_tmp = webstring[len(start_name):]
	       
		stringList = re.split('%|EUR-',_tmp)

		for k in range(len(stringList)):
			if len(stringList[k]) > 1:
				if stringList[k][0].isdigit() == False:
				
					Ind = re.search("\d",stringList[k])

					if Ind is None:
						PeerList.append(stringList[k][0:])
					else:
						PeerList.append(stringList[k][0:Ind.start()])

					substring = re.sub("[\(\[].*?[\)\]]", "", PeerList[-1])
					if substring[-1] == ' ':
						substring = substring[:-1]
	                
					PeerURLList.append('http://www.finanzen.net/aktien/'+ substring.replace('  ',' ').replace(' ','_').replace('.','')+'-Aktie')

		if len(PeerList) == 0:
			PeerList.append(None)
			PeerURLList.append(None)

		return PeerList,PeerURLList

	def _find_str(self,s, char,return_ind='end'):

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


	def _URL_online(self,URL):
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

