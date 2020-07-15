import pandas as pd
import random
class data_Processor(object):
	def __init__(self):
		self.train_Set = dict()
		self.test_Set = dict()

	def getDataset(self,path = r'C:\\Users\lm\Desktop\recommendsystem\code\CF\dataset\ml-1m\\ratings.dat'):
		'''
		:param path: file path
		:return: dataset
		'''
		dataset = []
		try:
			f = open(path,encoding='utf-8')
			for line in f:
				s = line.strip().split('::')
				if len(s) < 4:
					continue
				#print(s)
				dataset.append(s)
		except Exception as error:
			print("File error")
		return dataset


	def splitData(self,k,M,seed):
		'''
		:param M: divide into M slices
		:param k: the testset takes k slices of M
		:param seed: random seed
		:return:
		'''
		dataset = self.getDataset()
		random.seed(seed)
		for record in dataset:
			#print(random.randint(0, M))
			#print(self.test_Set)
			if random.randint(0,M) == k:
				self.test_Set.setdefault(record[0],{})
				self.test_Set[record[0]][record[1]] = record[2]
			else:
				self.train_Set.setdefault(record[0],{})
				self.train_Set[record[0]][record[1]] = record[2]

if __name__ == '__main__':
	dp = data_Processor()
	dp.getDataset()
	dp.splitData(M=8,k=3,seed=47)

	print(len(dp.train_Set.keys()))
	print(len(dp.test_Set.keys()))