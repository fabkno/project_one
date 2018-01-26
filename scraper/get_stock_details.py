import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import os
import re
import shutil,sys


class UpdateStockDetails(object):
	'''

	This classs is designed to update all stock information (usually not on a daily base) execept for the daily stock values

	
	'''
	def __init__(self,Path=None):

		self.today = datetime.datetime.today().date()

		if Path is None:
			self.Path = os.getcwd()+'/'
		else:
			self.Path = Path
		
		if os.path.isfile(self.Path + 'ListOfCompaniesFromIndices.p') == True:
			self.ListOfCompanies = pd.read_pickle(self.Path + 'ListOfCompaniesFromIndices.p')

			#add new columns
			for category in ['ISIN','WKN','SymbolFinanzen.net','Country','ListedIndizes','PeerGroupFinanzenNetName','PeerGroupFinanzenNetURLs','PeerGroupFinanzenNetOnline']:

				if category not in self.ListOfCompanies.keys() and category in ['ListedIndizes','PeerGroupFinanzenNetName','PeerGroupFinanzenNetURLs','PeerGroupFinanzenNetOnline']:
					self.ListOfCompanies[category] = pd.Series(np.nan,index=self.ListOfCompanies.index,dtype=object)

				elif category not in self.ListOfCompanies.keys() and category not in ['ListedIndizes','PeerGroupFinanzenNetName','PeerGroupFinanzenNetURLs','PeerGroupFinanzenNetOnline']:
					self.ListOfCompanies[category] = pd.Series(np.nan,index=self.ListOfCompanies.index)

		# To do check time stamp
		elif os.path.isfile(self.Path +'CompaniesByBranches.p') == True:
			self.CompaniesByBranches = pd.read_pickle(self.Path + 'CompaniesByBranches.p')


	
	def retrieve_company_list_from_indices(self):
		'''
		retrieves preliminary list of stocks include in list of indices

		default is given by DAX,MDAX,TECDAX,SDAX, EUROSTOXX, NASDAQ 100

		'''
		def _get_names_n_urls(html_object):
			'''
			finds names and urls of given html_object from finanzen.net

			Parameters
			------------

			html_object 

			Returns
			------------
			names : list of strings company names included in given index

			urls : list of strings correspond finanzen.net urls

			'''
			allLinks = list(html_object.findAll("table", {"class": "table"}))
			names = None
			urls = None
			for link in allLinks:
				if '"/aktien' in str(link) and 'NameISINLetzterVortagTiefHoch' in link.get_text():
					urls= ['http://www.finanzen.net'+a.get('href') for a in list(link.findAll('a')) if '"/aktien' in str(a)]
					names =[a.get_text() for a in list(link.findAll('a')) if '"/aktien' in str(a)]
			return names,urls

		#check if company file exists, if yes make copy and add date of copy
		if os.path.isfile(self.Path + 'ListOfCompaniesFromIndices.p'):
			shutil.copy(self.Path + 'ListOfCompaniesFromIndices.p',self.Path + 'ListOfCompaniesFromIndices_'+str(self.today)+'.p')

		#initialize new data frame
		ListOfCompaniesFromIndices = pd.DataFrame(columns=['Name','URL'])

		ListOfIndexUrls = ['https://www.finanzen.net/index/DAX/Werte',
				'https://www.finanzen.net/index/MDAX/Werte',
				'https://www.finanzen.net/index/SDAX/Werte',
				'https://www.finanzen.net/index/Dow_Jones/Werte',
				'https://www.finanzen.net/index/Nasdaq_100/Werte',
				'https://www.finanzen.net/index/Euro_Stoxx_50/Werte',
				'https://www.finanzen.net/index/TECDAX/Werte']
		#loop through all urls
		for IndexUrl in ListOfIndexUrls:
			
			if self._URL_online(IndexUrl) == False:
				print IndexUrl 
				raise ValueError('URL does not exist')

			page = requests.get(IndexUrl)
			soup = BeautifulSoup(page.content, 'html.parser')

			names,urls = _get_names_n_urls(soup)
	
			FirstCompany = names[0]

			if names is not None:
		
				if np.mod(len(names),2) == 0:

					for i in range(0,len(names),2):
						ListOfCompaniesFromIndices = ListOfCompaniesFromIndices.append({'Name':names[i],'URL':urls[i]},ignore_index=True)
				else:
					"check results for index", IndexUrl
			
			else:
				print IndexUrl

			#check if index website has multiple pages of companies
			for nextPage in range(2,100):
				_page = requests.get(IndexUrl+'@intpagenr_'+str(nextPage))
				_soup = BeautifulSoup(_page.content,'html.parser')
				names,urls = _get_names_n_urls(_soup)
				
				if names is not None:
					
					if np.mod(len(names),2) == 0:
						for i in range(0,len(names),2):
							ListOfCompaniesFromIndices = ListOfCompaniesFromIndices.append({'Name':names[i],'URL':urls[i]},ignore_index=True)
					else:
						"check results for index", IndexUrl
				
					if names[0] == FirstCompany:
						break

				if names is None:
					
					break

		ListOfCompaniesFromIndices.to_pickle(self.Path + 'ListOfCompaniesFromIndices.p')			



	def add_stock_details(self):

		'''
		fix bug country "letzte dividende"
		'''

		#shutil.copy(self.Path+'ListOfCompaniesFromIndices.p',self.Path+'ListOfCompaniesFromIndices_old.p')

		N=0
		
		while N <len(self.ListOfCompanies):
		#while N <16:
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

		
			if np.mod(N,100) == 0:
				self.ListOfCompanies.index.name = self.today
				self.ListOfCompanies.to_pickle(self.Path +'ListOfCompanies.p')
				print "saved"
			N+=1
		self.ListOfCompanies.index.name = self.today
		self.ListOfCompanies.to_pickle(self.Path +'ListOfCompanies.p')
		print "ListOfCompanies is saved"

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
		wkn = None
		isin = None

		soup = BeautifulSoup(page.content, 'html.parser')

		allIdentifiers= list(soup.findAll('title'))
		identifier= allIdentifiers[0].get_text().split()[-1].split(',')

		
		
		#if identifier[0][1:5].isdigit() == True:
		wkn = identifier[0][1:]

			

		if len(identifier) == 2:
			isin = identifier[1][:-1]
		elif len(identifier) == 3:
			isin = identifier[2][:-1]			
			symbol = identifier[1]
		#else:
			#isin = identifier[0][1:-1]

	
		#find country of company and all stock indices that list stock
		allIndices = list(soup.findAll("div", {"class": "box"}))

		# # prepare string to find peer group information
		# __tmp = url.rsplit('/', 1)[-1]

		# __tmp2 = " ".join(re.split('_|-',__tmp))

		# #capitalize first word
		# __Listtmp2 = __tmp2.split()
		# __Listtmp2[0] = __Listtmp2[0].upper()

		# string4PeerGroup = "Peer Group " + __tmp2
		# string4PeerGroup2 = "Peer Group " + ' '.join(__Listtmp2)

		# string4PeerGroup= string4PeerGroup.encode('latin1')
		# string4PeerGroup2= string4PeerGroup2.encode('latin1')

	
		PeerList = []
		PeerURLList = []
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
				
				_end_pos = self._find_str(zumUnternehmen,'Branchen')
				
				#if string "Branchen" not found look for string "Letzte" instead
				if _end_pos == -1:
					_end_pos = self._find_str(zumUnternehmen,'Letzte')

					# if string "Letzte" not found "LandXXX" is last sequence in string zumUnternehmen
					if _end_pos ==-1:
						_end_pos = None
					else:
						_end_pos -= 7
					
						

				else:
					_end_pos -= 9

				#Land = zumUnternehmen[self._find_str(zumUnternehmen,'Land')-1:self._find_str(zumUnternehmen,'Branchen')-9]
				Land = zumUnternehmen[self._find_str(zumUnternehmen,'Land')-1:_end_pos]
			elif "Peer Group " in str(name):
			#elif string4PeerGroup in str(name) or string4PeerGroup2 in str(name):

				_arefs= list(name.findAll('a',href=True))

				PeerList,PeerURLList = self._get_peer_company_names(_arefs,PeerList,PeerURLList)
				
		
		# print out single found features
		if  1 == 0:
			print "WKN " ,wkn
			print "ISIN" , isin
			print 'SymbolFinanzen.net ', symbol
			print 'Country ', Land
			print 'ListedIndizes', ListedIndizes
			print 'PeerGroupFinanzenNetName', PeerList

		if len(PeerList) ==0:
				PeerList = None
		if len(PeerURLList) == 0:
				PeerURLList = None
		return {'WKN':wkn,'ISIN':isin,'SymbolFinanzen.net':symbol,'Country':Land,'ListedIndizes':ListedIndizes,'PeerGroupFinanzenNetName':PeerList,'PeerGroupFinanzenNetURLs':PeerURLList}


	def _get_peer_company_names(self,ListOfStrings,PeerList,PeerURLList):

		'''
		helper function that finds from ListOfStrings the finanzen.net url and stock name
		Parameters
		-------------
		'''

	
		for _link in ListOfStrings:
			if '"/aktien/' in str(_link):
				
				PeerList.append(_link.get_text())
				PeerURLList.append('http://www.finanzen.net'+_link.get('href'))


		
		return PeerList,PeerURLList


# test = "Peer Group "

# for name in allIndices:
#     if test in str(name):

#         tmp= list(name.findAll('a',href=True))
        
#         for k,link in enumerate(tmp):
#             if '"/aktien/' in str(link):
            
#                 print link.get_text(), k
           
#                 print 'http://www.finanzen.net'+link.get('href')


	def _get_peer_company_names2(self,webstring,start_name):

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

