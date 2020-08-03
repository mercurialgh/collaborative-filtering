import sys
sys.path.append(r'C:\\Users\lm\Desktop\recommendsystem\code\CF')
from data_Process import data_Processor
import math
import operator

class userCF(object):
	def __init__(self):
		self.train = dict()
		self.test = dict()
		self.usersim = dict()

	def initData(self, k=3, M=8, seed=47):
		dp = data_Processor()
		dp.getDataset()
		dp.splitData(k, M, seed)
		self.train = dp.train_Set
		self.test = dp.test_Set

	def userSimilarity_1(self, train):
		"""
		one method for implementing user sim,on page 45-46.time complexity:n squared.USELESS!
		"""
		#self.usersim = dict()
		for u in train.keys():
			for v in train.keys():
				if u == v:
					continue
				self.usersim.setdefault(u,{})
				self.usersim[u][v] = len(set(train[u].keys()) & set(train[v].keys()))
				self.usersim[u][v] /= math.sqrt(len(set(train[u].keys())*len(set(train[v].keys()))))

	def userSimilarity_2(self, train):
		"""
		the other method of getting user similarity which is better than above on page 46
		"""
		item_user = dict()
		for user,items in train.items():
			for item in items.keys():
				if item not in item_user:
					item_user[item] = set()
				item_user[item].add(user)
		# calculate co-rated items for users
		N = dict()
		C = dict()
		for item,users in item_user.items():
			for u in users:
				N[u] = N.get(u,0)+1
				for v in users:
					#print(u,v)
					if u == v:
						continue
					C.setdefault(u,{})
					C[u].setdefault(v,0)
					C[u][v] += 1
		# calculate final similarity
		for u,related_users in C.items():
			self.usersim.setdefault(u,{})
			for v,cuv in related_users.items():
				self.usersim[u].setdefault(v,0)
				self.usersim[u][v] = cuv/math.sqrt(N[u]*N[v])

	def userSimilarity_3(self, train):
		"""
		the other method of getting user similarity which is better than above on page 49
		"""
		item_user = dict()
		for user,items in train.items():
			for item in items.keys():
				if item not in item_user:
					item_user[item] = set()
				item_user[item].add(user)
		# calculate co-rated items for users
		N = dict()
		C = dict()
		for item,users in item_user.items():
			for u in users:
				N[u] = N.get(u,0)+1
				for v in users:
					# print(u,v)
					if u == v:
						continue
					C.setdefault(u,{})
					C[u].setdefault(v,0)
					# Penalize the top items
					C[u][v] += 1/math.log(1+len(users),2)
		# calculate final similarity
		for u,related_users in C.items():
			self.usersim.setdefault(u,{})
			for v,cuv in related_users.items():
				self.usersim[u].setdefault(v,0)
				self.usersim[u][v] = cuv/math.sqrt(N[u]*N[v])

	def recommend(self, user, k=8, nitems=10):
		"""
		:param user: the user we recommend items to
		:param train: dataset
		:param k: most similar k users
		:param nitem: most similar n items
		:return: ranks for top k items
		"""
		rank = dict()
		interacteditems = self.train.get(user)
		for v,sim in sorted(self.usersim[user].items(),key=operator.itemgetter(1),reverse=True)[0:k+1]:
			for i,rate in self.train[v].items():
				if i in interacteditems:
					continue
				rank.setdefault(i,0)
				rank[i] += sim * int(rate)
		return dict(sorted(rank.items(),key = lambda x:x[1],reverse = True)[0:nitems])

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
	uc = userCF()
	uc.initData(3,8,47)
	#print(len(uc.train.keys()))
	#print(uc.test)
	uc.userSimilarity_3(uc.train)
	k = 8
	nitems = 10
	recall,precision = uc.recallAndPrecision(k=k, nitems=nitems)
	coverage = uc.coverage(k=k, nitems=nitems)
	popularity = uc.popularity(k=k, nitems=nitems)
	print("recall: %.5f, precision: %.5f, coverage: %.5f, popularity: %.5f"\
		  %(recall, precision, coverage, popularity))