import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
import LDA
import spoken
import written
import PCA
import seaborn as sb
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true,y_pred,address):
	cf_mat = confusion_matrix(y_true,y_pred)
	sb.heatmap(cf_mat/np.sum(cf_mat),annot=True,fmt='.2%',cmap='Blues')
	plt.savefig(address)

# normalize the data given in form of a map 
def normalize(data):
	total = []
	for i in data.keys():
		for j in data[i]:
			if total == []:
				for k in j :
					total.append([k])
			else:
				for k in range(len(j)):
					total[k].append(j[k])
	
	total = np.asarray(total)
	ans = {}
	for i in data.keys():
		if not i in ans.keys():
			ans[i]=[]
		for j in data[i]:
			temp = []
			for k in range(len(j)):
				temp.append((j[k]-np.average(total[k]))/np.std(total[k]))
			ans[i].append(temp)
	
	return ans
			

# to get euclidian distance
def euclid_dist(a,b):
	ans=0
	for i in range(len(a)):
		ans+=pow((a[i]-b[i]),2);
	return pow(ans,1/2)

def KNN(train_map,dev_map,K,address):
	Predicted = []
	actual = []
	total = 0
	correct = 0
	f = open(address,"w")
	for i in dev_map.keys():
		for j in dev_map[i]:
			distances = []
			for k in train_map.keys():
				for l in train_map[k]:
					distances.append([euclid_dist(j,l),k])
			distances.sort(key = lambda x :x[0])
			temp = {}
			for ii in dev_map.keys():
				temp[ii]=0
			for m in range(K):
				if not distances[m][1] in temp.keys() :
					temp[distances[m][1]]=0
				temp[distances[m][1]]+=1
			sorted_temp = dict(sorted(temp.items(),key = lambda x : -1*x[1]))
			predicted = -1
			for m in sorted_temp.keys():
				predicted = m
				break
			if(predicted == i):
				correct+=1
			total+=1
			f.write(str(i)+" ")
			for ii in dev_map.keys():
				f.write(str(temp[ii]/K)+" ")
			
			Predicted.append(predicted)
			actual.append(i)
			f.write("\n")
	print("accuracy = "+str(correct*100/total)+" for k = "+str(K))
	return actual,Predicted


train = open("train.txt","r")
synthetic_data_train_map = {}
for i in train.readlines():
	feature = list(map(float,i[:-1].split(",")[:-1]))
	given_class = float(i[:-1].split(",")[-1])
	if not given_class in synthetic_data_train_map.keys() :
		synthetic_data_train_map[given_class]=[]
	synthetic_data_train_map[given_class].append(feature)

dev = open("dev.txt","r")
synthetic_data_dev_map = {}
for i in dev.readlines():
	feature = list(map(float,i[:-1].split(",")[:-1]))
	given_class = float(i[:-1].split(",")[-1])
	if not given_class in synthetic_data_dev_map.keys() :
		synthetic_data_dev_map[given_class]=[]
	synthetic_data_dev_map[given_class].append(feature)

synthetic_data_dev_map = normalize(synthetic_data_dev_map)
synthetic_data_train_map=normalize(synthetic_data_train_map)

# print("synthetic data")
# for k in [15]:
# 	a,c=PCA.pca(synthetic_data_train_map,1)
# 	b = PCA.reducedPCA(synthetic_data_dev_map,c)
# 	actual,Predicted = KNN(a,b,k,"synthetic_knn_scores.txt")
# 	plot_confusion(actual,Predicted,"confusion_matrix_knn_synthetic.jpg")


path = "/mnt/c/Users/srava/Desktop/sem6/PRML - CS5691/assignment4/"
folders = [path+"coast",path+"forest",path+"highway",path+"mountain",path+"opencountry"]
image_data_train = {}
image_data_dev = {}

for id,i in enumerate(folders) :
	image_data_train[id+1]=[]
	image_data_dev[id+1]=[]
	for j in os.listdir(i+"/train/"):
		temp_file=open(i+"/train/"+j,"r")
		temp = []
		for idx,k in enumerate(temp_file.readlines()):
			l = list(map(float,k[:-1].split(" ")))
			temp.extend(l)
		image_data_train[id+1].append(temp)
	for j in os.listdir(i+"/dev/"):
		temp_file=open(i+"/dev/"+j,"r")
		temp=[]
		for idx,k in enumerate(temp_file.readlines()):
			l = list(map(float,k[:-1].split(" ")))
			temp.extend(l)
		image_data_dev[id+1].append(temp)

image_data_dev = normalize(image_data_dev)
image_data_train = normalize(image_data_train)
print("written data")
for k in [10]:
	a,b,_,_,_,_=written.convert()
	for dim in [10]:
		print(dim)
		a,c = PCA.pca(a,dim)
		b=PCA.reducedPCA(b,c)
		actual , Predicted = KNN(a,b,k,"written_knn_scores.txt")
		plot_confusion(actual,Predicted,"confusion_matrix_knn_written.jpg")
print("spoken data")
for k in [15]:
	a,b,_,_,_,_=spoken.convert()
	for dim in [10]:
		print(dim)
		a,c = PCA.pca(a,dim)
		b = PCA.reducedPCA(b,c)
		actual , Predicted = KNN(a,b,k,"spoken_knn_scores.txt")
		plot_confusion(actual,Predicted,"confusion_matrix_knn_spoken.jpg")
# print("image data")
# for k in [10]:
# 	for d in [20,35,50,70,80,100]:
# 		a,c = PCA.pca(image_data_train,d)
# 		b = PCA.reducedPCA(image_data_dev,c)
# 		actual,Predicted = KNN(a,b,k,"image_knn_scores.txt")
# 		plot_confusion(actual,Predicted,"confusion_matrix_knn_image.jpg")
	
