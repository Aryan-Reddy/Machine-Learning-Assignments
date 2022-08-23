import enum
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

def lda(data_dict):
	total_mean = []
	total_points = 0
	S_w=[]
	all_means =[]
	# first calculate within class scatter 
	for i in data_dict.keys():
		mean = np.array(data_dict[i][0])
		for idx in range(1,len(data_dict[i])):
			mean += np.array(data_dict[i][idx])
		
		if(total_mean==[]):
			total_mean=mean
		else:
			total_mean+=mean

		total_points+=len(data_dict[i])

		for idx in range(len(mean)):
			mean[idx]/=len(data_dict[i])
		
		all_means.append(mean)

		x=np.array(data_dict[i][0])
		temp = x-mean
		temp = np.resize(temp,(len(temp),1))
		S_i = (temp)@(np.transpose(temp))
		for idx in range(1,len(data_dict[i])):
			x=np.array(data_dict[i][idx])
			temp = x-mean
			temp = np.resize(temp,(len(temp),1))
			S_i += (temp)@(np.transpose(temp))
		if(S_w==[]):
			S_w=S_i
		else:
			S_w+=S_i

	# calculate between class scatter 
	S_b = []
	for i in range(len(total_mean)):
		total_mean[i]/=total_points

	for i,j  in enumerate(data_dict.keys()):
		temp = all_means[i]-total_mean
		temp = np.resize(temp,(len(temp),1))
		if(S_b==[]):
			S_b = ((temp)@(np.transpose(temp)))
		else:
			S_b+=(temp)@(np.transpose(temp))

	# print (S_w)
	# print("#######################")
	# print(S_b)

	e,v = np.linalg.eig(np.linalg.inv(S_w)@S_b)
	ind = list(range(len(e)))
	ind.sort(key=lambda i:e[i],reverse=True)
	reducedData = {}
	for i in data_dict.keys():
		reducedData[i]=[]
		for j in range(len(data_dict[i])):
			temp = []
			for k in range(len(data_dict.keys())-1):
				ind2 = ind[k]
				temp.append(np.dot(v[:,ind2],data_dict[i][j]))
			reducedData[i].append(temp)
			
	return reducedData

