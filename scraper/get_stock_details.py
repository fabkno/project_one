import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

def find_str(s, char):

	'''
	finds position of substring in string returns position of last found character 

	Parameters
	----------------

	s string : string in which to look for substring

	char string : substring 

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
					return index+len(char)+1

			index += 1

	return -1


def get_stock_details_from_url(URL):
	
	#check if URL exists
	page = requests.get(URL)
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
	for name in allIndices:
		if "Zur Aktie" in str(name):#Zum Unternehmen       
			zurAktie = name.get_text()
			num_ = find_str(zurAktie,"Indizes")

			if num_ != -1:
				ListedIndizes = zurAktie[num_:].split(',')
			else:
				ListedIndizes = None

		elif "Zum Unternehmen" in str(name):
			zumUnternehmen = name.get_text()
			Land = zumUnternehmen[find_str(zumUnternehmen,'Land')-1:find_str(zumUnternehmen,'Branchen')-9]

	return {'WKN':wkn,'ISIN':isin,'SymbolFinanzen.net':symbol,'Country':Land,'ListedIndizes':ListedIndizes}


data = pd.read_pickle('companies_by_branches.p')
companies = data.copy()
companies['WKN'] = pd.Series(np.nan,index=companies.index)
companies['ISIN'] = pd.Series(np.nan,index=companies.index)
companies['SymbolFinanzen.net'] = pd.Series(np.nan,index=companies.index)
companies['Country'] = pd.Series(np.nan,index=companies.index)
companies['ListedIndizes'] = pd.Series(np.nan,index=companies.index,dtype=object)

for i in range(25,40):
	
	details = get_stock_details_from_url(companies.loc[i]['URL'])
	
	companies.loc[i,['ISIN','WKN','SymbolFinanzen.net','Country']] = [details['ISIN'],details['WKN'],details['SymbolFinanzen.net'],details['Country']]
	companies.at[i,'ListedIndizes'] = details['ListedIndizes']
	
	print companies.loc[i][['Name','Country','SymbolFinanzen.net','ListedIndizes']]

companies.to_pickle('ListofCompanies.p')