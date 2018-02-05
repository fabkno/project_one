import numpy as np
import inspect
import time
import datetime
import os
class Log(object):
	"""
	simple logfile class creates a logfile that includes the time / class of origin and method in which log file was called

	Parameters
	-----------------

	LogFileName : string (default = None ), name of logfile, if None Name will contain current date

	PathData : string (default = None), Path for folder ../data/

	Attributes
	-----------------

	today : datetime object, current day

	LogFileName : string, name of the logfile

	LogFilePath : string, full path in which logfile is saved

	PathData : string, path for until folder "../data/"


	Methods
	-----------------

	logging: opens and writes message into logfile


	"""
	def __init__(self,LogFileName=None,PathData = None):

		self.today = datetime.datetime.today().date()
		if PathData is None:
			self.PathData = os.path.dirname(os.getcwd())+'/data/'
		else: self.PathData = PathData

		if LogFileName is None:
			self.LogFileName = str(self.today.year)+'_'+str(self.today.month)+'_'+str(self.today.day)
		else: 
			self.LogFileName = LogFileName

		if os.path.exists(self.PathData + 'logs/') is False:
			os.makedirs(self.PathData + 'logs/')

		self.LogFilePath = self.PathData + 'logs/' + self.LogFileName+'.txt'

	def logging(self,text):

		with open(self.LogFilePath,'a') as f:
			_time = str(time.strftime("%d %b %Y %H:%M:%S",time.localtime())) 
			string = " ---> class: "+self.__class__.__name__+ "  method: "+inspect.stack()[1][0].f_code.co_name
			f.write(_time+string+ "\t>>> "+text+"\n")



