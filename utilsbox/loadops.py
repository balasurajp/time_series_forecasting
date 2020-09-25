import __init__
import os, logging, joblib
import numpy as np, pandas as pd #, modin.pandas as pd
from utilsbox.readops import NormalizeColumns, ProcessTimestamp, SupervisedSampler, reading_default, combine_databook

class PowerTAC:
	def __init__(self, config, mode):
		self.paths = config['paths']
		self.params = config['params']
		self.ndesign = config['algoparams']['netdesign']
		self.read_data(datamode=mode)
		print("PowerTAC Dataloader!!")

	def read_data(self, datamode):
		if self.paths['type']=='storage':
			if os.path.exists(f"/tmp/PowerTAC_{self.params['identity']}_{datamode}book.pkl"):
				logging.info(f"Reading data from tmp fileStorage")
				print("Reading data from tmp fileStorage")
				self.databook = joblib.load(f"/tmp/PowerTAC_{self.params['identity']}_{datamode}book.pkl")
			else:
				logging.info(f"Reading data from data fileStorage")
				print("Reading data from data fileStorage")
				self.databook = reading_default(self.paths, self.params, encodingtime='base2encode', mode=datamode)
				# joblib.dump(self.databook, f"/tmp/PowerTAC_{self.params['identity']}_{datamode}book.pkl")
	
		elif self.paths['type']=='mongo':
			raise NotImplementedError('must implement mongodb retrieval')
	
	def get_databook(self):
		return self.databook
	
	def get_samples(self):
		samplebook = combine_databook(self.databook)
		return samplebook
	
	# def get_scalebook(self):
	# 	scalebook = joblib.load(f"{self.paths['data']}/scalebook.pkl")
	# 	return scalebook