import sys
sys.path.append(r'C:\\Users\lm\Desktop\recommendsystem\code\CF')
from data_Process import data_Processor
import random
import operator
import math
import numpy as np
class itemCF(object):
	def __init__(self):
		self.train = dict()
		self.test = dict()
		self.itemSim = dict()
	def initDataset(self,k=3,M=8,seed=47):
		dp = data_Processor()
		dp.splitData(k=k,M=M,seed=seed)
		self.train, self.test = dp.train_Set, dp.test_Set

	def itemSimilarity_1(self):
		item_Users = dict()
		for user,items in self.train.items():
			for item,rating in items.items():
				if item not in item_Users:
					item_Users.setdefault(item,set())
				item_Users[item].add(user)
		C = dict()
		N = dict()
		for i1 in item_Users.keys():
			C.setdefault(i1,{})
			N[i1] = N.get(i1,0) + len(item_Users[i1])
			for i2 in item_Users.keys():
				C[i1].setdefault(i1,0)
				if i1 == i2:
					continue
				C[i1][i2] = len(item_Users[i1])&len(item_Users[i2])
				C[i1][i2] /= math.sqrt(len(item_Users[i1])*len(item_Users[i2]))
		self.itemSim = C

	def itemSimilarity_2(self):
		"""
		on page 53:get item similarity by user-item table
		"""
		N = dict()
		C = dict()
		# get C[i][j]
		for user,items in self.train.items():
			for i in items.keys():
				N[i] = N.get(i, 0) + 1
				C.setdefault(i, {})
				for j in items.keys():
					C[i].setdefault(j, 0)
					if i == j:
						continue
					C[i][j] += 1
		# calculate final similarity between items
		for i,related_items in C.items():
			self.itemSim.setdefault(i, {})
			for j,cij in related_items.items():
				self.itemSim[i].setdefault(j, 0)
				self.itemSim[i][j] += cij / math.sqrt(N[i] * N[j])

	def itemSimilarity_3(self):
		"""
		on page 58:get item similarity by user-item table with active user punishment
		"""
		N = dict()
		C = dict()
		# get C[i][j]
		for user,items in self.train.items():
			for i in items.keys():
				N[i] = N.get(i, 0) + 1
				C.setdefault(i, {})
				for j in items.keys():
					C[i].setdefault(j, 0)
					if i == j:
						continue
					# penalize the active user
					C[i][j] += 1 / math.log(1 + len(items) * 1)
		# calculate final similarity between items
		for i,related_items in C.items():
			self.itemSim.setdefault(i, {})
			for j,cij in related_items.items():
				self.itemSim[i].setdefault(j, 0)
				self.itemSim[i][j] += cij / math.sqrt(N[i] * N[j])

	def itemSimilarity_3_Norm(self):
		"""
		on page 58:get item similarity by user-item table with active user punishment
		"""
		N = dict()
		C = dict()
		# get C[i][j]
		for user,items in self.train.items():
			for i in items.keys():
				N[i] = N.get(i, 0) + 1
				C.setdefault(i, {})
				for j in items.keys():
					C[i].setdefault(j, 0)
					if i == j:
						continue
					# penalize the active user
					C[i][j] += 1 / math.log(1 + len(items) * 1)
		# calculate final similarity between items
		for i,related_items in C.items():
			self.itemSim.setdefault(i, {})
			for j,cij in related_items.items():
				self.itemSim[i].setdefault(j, 0)
				self.itemSim[i][j] += cij / math.sqrt(N[i] * N[j])

		for i1,items in self.itemSim.items():
			maxsim = 0
			for i2,sim in items.items():
				maxsim = max(maxsim,sim)
			for i2, sim in items.items():
				self.itemSim[i1][i2] /= maxsim


	def recommend(self,user,k = 8,nitems = 10):
		"""
		:param user: the user recommending items for
		:param k: most similar k items
		:param nitems: number of items to recommend
		:return: most top nitems items and corresponding rank
		"""
		rank = dict()
		ru = self.train[user]
		for item_u,rating in ru.items():
			for item_v,sim in sorted(self.itemSim[item_u].items(),key=operator.itemgetter(1),reverse=True)[0:k+1]:
				if item_v in ru:
					continue
				rank.setdefault(item_v, 0)
				rank[item_v] += int(rating) * sim
		return dict(sorted(rank.items(), key=operator.itemgetter(1),reverse=True)[0:nitems])

	def recallAndPrecision(self, k=8, nitems=10):
		"""
		check on page 43
		:param k:top k similar users
		:param nitems:top n similar items
		:return: recall,precision
		"""
		hit = 0
		recall = 0
		precision = 0
		for user in self.train.keys():
			tu = self.test.get(user, {})
			rank = self.recommend(user, k, nitems)
			for item,pui in rank.items():
				if item in tu:
					hit += 1
			recall += len(tu)
			precision += nitems
			#print(tu,rank,recall,precision)
		return hit/(recall*1.0), hit/(precision*1.0)

	def coverage(self, k=8, nitems=10):
		"""
		check on page 43
		:return: coverage of recommended items
		"""
		all_items = set()
		recommended_items = set()
		for user in self.train.keys():
			for item in self.train[user].keys():
				all_items.add(item)
			rank = self.recommend(user, k=k, nitems=nitems)
			for item,pvi in rank.items():
				recommended_items.add(item)
		return len(recommended_items)/(len(all_items) * 1.0)

	def popularity(self, k=8, nitems=10):
		"""
		check on page 44
		:return: average popularity of recommended items
		"""
		item_popularity = dict()
		for user in self.train.keys():
			for item in self.train[user].keys():
				if item not in item_popularity:
					item_popularity[item] = 0
				item_popularity[item] += 1
		ret = 0
		n = 0
		for user in self.test.keys():
			rank = self.recommend(user, k=k, nitems=nitems)
			for item,pui in rank.items():
				ret += math.log(1+item_popularity[item])
				n += 1
		ret /= n * 1.0
		return ret

if __name__ == '__main__':
	ic = itemCF()
	ic.initDataset(k=3,M=8,seed=47)
	ic.itemSimilarity_3_Norm()
	k = 8
	nitems = 10
	recall, precision = ic.recallAndPrecision(k=k,nitems=nitems)
	coverage = ic.coverage(k=k,nitems=nitems)
	popularity = ic.popularity(k=k,nitems=nitems)
	print("recall: %.5f, precision: %.5f, coverage: %.5f, popularity: %.5f" \
		  % (recall, precision, coverage, popularity))



