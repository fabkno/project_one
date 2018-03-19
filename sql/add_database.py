import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime
import sqlite3 as sql
import os,sys


def get_tuple(data):
	new = [single for single in data['ListedIndizes'].split('\'') if single[0] not in ["[","u",",","]"]]
	stock_indices = [new[0]] + [_str[1:] for _str in new[1:]]

	return tuple([unicode(data.values[0].decode('UTF-8')),data.values[6],data.values[2],data.values[3]]) +tuple(stock_indices[0:4])


def add_stock_info(PathData = None):

	if PathData is None:
		PathData = os.path.dirname(os.getcwd())+'/data/'

	db = sql.connect(PathData+'predictions/stocks/predictions.db')

	cursor = db.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

	tableNames = [i[0] for i in cursor.fetchall()]
	
	if "stockinfo" not in tableNames == False:
		
		db.cursor().execute(''' CREATE TABLE stockinfo(id INTEGER PRIMARY KEY, Name TEXT, Label TEXT, ISIN TEXT, WKN TEXT, Index1 TEXT, Index2 TEXT, Index3 TEXT,Index4 TEXT)''')
		db.commit()

		l = pd.read_csv(PathData+'company_lists/full_list_w_indices.csv',index_col=0)

		for i in range(len(l)):
			datatuple= get_tuple(l.loc[i])
			if len(datatuple) <8 :
				datatuple = datatuple + tuple([None for i in range(8-len(datatuple))])

			db.cursor().execute('''INSERT INTO stockinfo(Name,Label,ISIN,WKN,Index1,Index2,Index3,Index4) VALUES(?,?,?,?,?,?,?,?)''', datatuple)

		db.commit()
		db.close()
		print("stockinfo database successfully added")

	else:
		print("Table stockinfo does already exist")


def update_shorts(ListOfLabels='all',ListOfPredictionDays='all',PathData=None):

	def check_if_entry_exists(db,data,Label):
	#check if entry exists in database

		cursor = db.cursor()
		
		Date = data['Date'].date()

		Position = data[Label]
		
		x = db.cursor().execute('''SELECT Label,Date FROM shorts WHERE Label=? AND ShortPosition = ? AND Date=?''',(Label,Position,Date))
		tmp = x.fetchall()
		
		
		if len(tmp) == 0:
			return False

		elif len(tmp) >0 and Position == tmp[0][1]:
		
			return "update"

		else:
			return True

	
	def add_entry(db,pandasDB,label):

		datatuple=  tuple([label,float(pandasDB[label]),pandasDB['Date'].date()])
		
		db.cursor().execute(''' INSERT INTO shorts(Label,ShortPosition,Date) VALUES(?,?,?)''',datatuple)
		db.commit()

	def update_entry(db,pandasDB,label):

		datatuple = tuple([label,float(pandasDB[label]),pandasDB['Date'].date()])
		
		db.cursor().execute(''' UPDATE shorts SET ShortPosition = ? WHERE Label = ? and Date = ? ''',(datatuple[1],datatuple[0],datatuple[2]))
		db.commit()


	if PathData is None:
		PathData = os.path.dirname(os.getcwd())+'/data/'

	db = sql.connect(PathData+'predictions/stocks/predictions.db')

	cursor = db.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

	tableNames = [i[0] for i in cursor.fetchall()]
	

	if "shorts" not in tableNames:

		cursor.execute(''' CREATE TABLE shorts(id INTEGER PRIMARY KEY, Label TEXT, ShortPosition FLOAT, Date DOB)''')
		db.commit()

	data = pd.read_pickle(PathData+'shorts/list.p')

	if ListOfLabels == 'all':
		labels = data.columns[1:]
	else:
		labels = ListOfLabels

	for label in labels:
		print label
		df = data.loc[:,data.columns.isin([label,'Date'])]
		if ListOfPredictionDays is not 'all':
			dff = df.loc[df['Date'].isin(ListOfPredictionDays)]

		else:
			dff = df
		
		for n in dff.index.tolist():
			
			status = check_if_entry_exists(db,dff.loc[n],label)
			
			if status == False:
				add_entry(db,dff.loc[n],label)
			
			elif status =='update':
				update_entry(db,dff.loc[n],label)
				
				

			
			


	db.close()


