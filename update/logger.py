import numpy as np
import inspect
import time
import datetime

class Log(object):

	def __init__(self,LogFileName=None,PathData = None):

		self.today = datetime.datetime.today().date()
		if PathData is None:
				self.PathData = os.path.dirname(os.getcwd())+'/data/'
		else: self.PathData = PathData

		if LogFileName is None:
			self.LogFileName = str(self.today.year)+'_'+str(self.today.month)+'_'+str(self.today.day)
		else: self.LogFileName = LogFileName

	def logging(self,text,className):

		with open(self.LogFileName,'a+') as f:
			string = str(time.strftime("%d %b %Y %H:%M:%S",time.localtime()))+" from class: "+className+ "method: "+inspect.stack()[1][0].f_code.co_name
			f.write(string+'\t'+text+'\n')



