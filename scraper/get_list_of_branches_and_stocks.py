import requests
from bs4 import BeautifulSoup
import pandas as pd


############### function header #################

def get_companies_for_branch(companies,branch_label,url):
            
    print branch_label
    _page = requests.get(url)
    _soup = BeautifulSoup(_page.content,'html.parser')
    _links = list(_soup.findAll('a',href=True))
    
    start = [k for k in range(len(_links)) if '/aktien/dividenden/?intHistorisch' in str(_links[k])][-1]
    end =  [k for k in range(len(_links)) if '/aktien/primestandard/alle' in str(_links[k])][0]
    
    #find url to stop : when companies repeat quit
    _first_company = None
    
    for k in range(start+1,end):
        if '/aktien/' in str(_links[k]):
            if _first_company is None:
                _first_company = _links[k].get_text()
            
            
            companies = companies.append({'Name':_links[k].get_text(),
                                          'URL':'http://www.finanzen.net'+_links[k].get('href'),
                                          'Branche':branch_label},ignore_index =True)
    
   
    for nextPage in range(2,100):
        
        _page = requests.get(url+'@intpagenr_'+str(nextPage))
        _soup = BeautifulSoup(_page.content,'html.parser')
        _links = list(_soup.findAll('a',href=True))
        
        start = [k for k in range(len(_links)) if '/aktien/dividenden/?intHistorisch' in str(_links[k])][-1]
        end =  [k for k in range(len(_links)) if '/aktien/primestandard/alle' in str(_links[k])][0]
        
        
        for k in range(start+1,end):
            if '/aktien/' in str(_links[k]):
                if _links[k].get_text() == _first_company:
                    print "page ", nextPage, " equal to first page"
                    return companies
                    
                else:
                    companies = companies.append({'Name':_links[k].get_text(),
                                          'URL':'http://www.finanzen.net'+_links[k].get('href'),
                                          'Branche':branch_label},ignore_index =True)
    
    print  ValueError('Scraper probably not correct')
    return companies


####### getting all branches @ finanzen.net ########

page = requests.get("https://www.finanzen.net/branchen/")
soup = BeautifulSoup(page.content, 'html.parser')

allLinks= list(soup.findAll('a'))

branchen = pd.DataFrame(columns=['Name','URL'])

for i in range(len(allLinks)):
    if '/branchen/' in str(allLinks[i]) and '/branchen/\"' not in str(allLinks[i]):
        branchen = branchen.append({'Name':allLinks[i].get_text(),'URL':'http://www.finanzen.net'+allLinks[i].get('href')},ignore_index=True)



################ get all companies belonging to all branches ################
companies = pd.DataFrame(columns=['Name','URL','Branche'])

for _num in range(len(branchen)):
    companies = get_companies_for_branch(companies,branchen['Name'][_num],branchen['URL'][_num])


######################## find companies assigned to more than one branch and merge branches #######################

companies_new = companies.copy()

doubleCompanies = companies_new.loc[companies_new['Name'].duplicated()]['Name'].values

for i in range(len(doubleCompanies)):
    ListOfBranches = companies.loc[companies['Name'] == doubleCompanies[i]]['Branche'].values
    companies_new.loc[companies_new['Name'] == doubleCompanies[i],'Branche'] = [ListOfBranches]

########## delete duplicated entries ###############
companies_new.drop_duplicates('Name',inplace=True)
companies_new.reset_index(drop=True,inplace=True)

print 'final number of companies is ', len(companies_new)

companies_new.to_pickle('companies_by_branches.p')