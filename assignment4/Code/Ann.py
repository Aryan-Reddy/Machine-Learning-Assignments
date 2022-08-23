import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import os
import random
import seaborn as sb
import matplotlib.pylab as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import spoken
def getMeanCov(data):
	n = len(data)
	f = len(data[0])
	mean = np.zeros((f,1))
	for i in range(n):
		for j in range(f):
			mean[j][0] += data[i][j]
	for j in range(f):
		mean[j][0] /= n
	cov = np.zeros((f,f))
	for i in range(n):
		tmp = np.zeros((f,1))
		for j in range(f):
			tmp[j][0] = data[i][j]
		diff = tmp - mean
		cov += diff @ np.transpose(diff)
	cov /= n
	return mean,cov
def PCA(data,ncomp):
	n = len(data)
	f = len(data[0])
	mean,cov = getMeanCov(data)
	e,v = np.linalg.eig(cov)
	ind = list(range(f))
	ind.sort(key=lambda i:e[i],reverse=True)
	reducedData = []
	qs = []
	for j in range(ncomp):
		qs.append(v[:,ind[j]])
	for i in range(n):
		reducedData.append([])
		for j in range(ncomp):
			ind2 = ind[j]
			vec = v[:,ind2]
			reducedData[-1].append(np.dot(vec,data[i]))
	return reducedData,qs
def lda(data,dataCls):
	n = len(data)
	data_dict = {}
	for i in range(n):
		if dataCls[i] in data_dict.keys():
			data_dict[dataCls[i]].append(data[i])
		else:
			data_dict[dataCls[i]] = [data[i]]
	total_mean = []
	total_points = 0
	S_w=[]
	all_means =[]
	# first calculate within class scatter 
	for i in data_dict.keys():
		mean = np.array(data_dict[i][0])
		for idx in range(1,len(data_dict[i])):
			mean += np.array(data_dict[i][idx])
		
		if(len(total_mean)==0):
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
		if(len(S_w)==0):
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
		if(len(S_b)==0):
			S_b = ((temp)@(np.transpose(temp)))
		else:
			S_b+=(temp)@(np.transpose(temp))

	# print (S_w)
	# print("#######################")
	# print(S_b)

	e,v = np.linalg.eig(np.linalg.inv(S_w)@S_b)
	ind = list(range(len(e)))
	ind.sort(key=lambda i:np.abs(e[i]),reverse=True)
	rdata = []
	rdataCls = []
	clsCnt = len(data_dict.keys())
	qs = []
	for j in range(clsCnt-1):
		qs.append(list(np.real(v[:,ind[j]])))
	for i in data_dict.keys():
		for j in range(len(data_dict[i])):
			temp = []
			for k in range(clsCnt-1):
				ind2 = ind[k]
				temp.append(np.real(np.dot(v[:,ind2],data_dict[i][j])))
			rdata.append(temp)
			rdataCls.append(i)
	return rdata,rdataCls,qs
def reducedPCA(data, qs):
	n = len(data)
	reducedData = []
	for i in range(n):
		reducedData.append([])
		for j in range(len(qs)):
			reducedData[-1].append(np.dot(data[i],qs[j]))
	return reducedData
def plot_confusion(y_true,y_pred,address):
	cf_mat = confusion_matrix(y_true,y_pred)
	sb.heatmap(cf_mat/np.sum(cf_mat),annot=True,fmt='.2%',cmap='Blues')
	plt.savefig(address)
def shuffle(data,dataCls):
	n = len(data)
	tmpInd = list(range(n))
	random.seed(10)
	random.shuffle(tmpInd)
	tmpData = data[:]
	tmpDataCls = dataCls[:]
	for i in range(n):
		data[i] = tmpData[tmpInd[i]]
		dataCls[i] = tmpDataCls[tmpInd[i]]
# trainFile = open("SyntheticData/train.txt",'r')
# writeFile = open("ANN.txt",'w')
# clsCnt = 2
# trainData = []
# trainDataCls = []
# for line in trainFile:
# 	words = line.split(',')
# 	n = len(words)
# 	c = (int(words[-1]) - 1)
# 	trainDataCls.append(c)
# 	trainData.append([])
# 	for i in range(n-1):
# 		trainData[-1].append(float(words[i]))
# shuffle(trainData,trainDataCls)
# #trainData,qs = PCA(trainData,1) #change here for feature
# #trainData,trainDataCls,qs2 = lda(trainData,trainDataCls)
# Annclf = MLPClassifier(random_state=1,hidden_layer_sizes=(70,),max_iter=500,activation='relu').fit(trainData,trainDataCls)	
# devFile = open("SyntheticData/dev.txt",'r')
# devData = []
# devDataCls = []
# for line in devFile:
# 	words = line.split(',')
# 	n = len(words)
# 	c = (int(words[-1]) - 1)
# 	devData.append([])
# 	devDataCls.append(c)
# 	for i in range(n-1):
# 		devData[-1].append(float(words[i]))
# #devData = reducedPCA(devData,qs)
# #devData = reducedPCA(devData,qs2)
# for i,d in enumerate(devData):
# 	writeFile.write(str(devDataCls[i]))
# 	writeFile.write(" ")
# 	probs = Annclf.predict_proba([devData[i]])[0]
# 	for j in range(clsCnt):
# 		writeFile.write(str(probs[j]))
# 		writeFile.write(" ")
# 	writeFile.write("\n")
# writeFile.close()
# print(Annclf.score(devData,devDataCls))


# imgTypes = ['coast','forest','highway','mountain','opencountry']
# clsCnt2 = len(imgTypes)
# trainData2 = []
# trainDataCls2 = []
# for curCls,imgType in enumerate(imgTypes):
# 	tmpDir = 'ImageData/'+ imgType + '/train'
# 	for example in os.listdir(tmpDir):
# 		trainFile2 = os.path.join(tmpDir,example)
# 		if not os.path.isfile(trainFile2):
# 			continue
# 		fptr = open(trainFile2,'r')
# 		trainData2.append([])
# 		trainDataCls2.append(curCls)
# 		for line in fptr:
# 			words = line.split(' ')
# 			for word in words:
# 				trainData2[-1].append(float(word))
# shuffle(trainData2,trainDataCls2)
# norm2 = MinMaxScaler().fit(trainData2)
# trainData2 = norm2.transform(trainData2)
# #trainData2,qs = PCA(trainData2,100) #change here for features
# #trainData2,trainDataCls2,qs2 = lda(trainData2,trainDataCls2)
# Annclf2 = MLPClassifier(hidden_layer_sizes=(33,12),activation='tanh',random_state=1,max_iter=400).fit(trainData2,trainDataCls2)
# devData2 = []
# devDataCls2 = []
# for curCls,imgType in enumerate(imgTypes):
# 	tmpDir = 'ImageData/' + imgType + '/dev'
# 	for example in os.listdir(tmpDir):
# 		devFile2 = os.path.join(tmpDir,example)
# 		if not os.path.isfile(devFile2):
# 			continue
# 		fptr = open(devFile2,'r')
# 		devData2.append([])
# 		devDataCls2.append(curCls)
# 		for line in fptr:
# 			words = line.split(' ')
# 			for word in words:
# 				devData2[-1].append(float(word))
# devData2 = norm2.transform(devData2)
# #devData2 = reducedPCA(devData2,qs)
# #devData2 = reducedPCA(devData2,qs2)
# for i,d in enumerate(devData2):
# 	writeFile.write(str(devDataCls2[i]))
# 	writeFile.write(" ")
# 	probs = Annclf2.predict_proba([devData2[i]])[0]
# 	for j in range(clsCnt2):
# 		writeFile.write(str(probs[j]))
# 		writeFile.write(" ")
# 	writeFile.write("\n")
# writeFile.close()
# devDataPred = Annclf2.predict(devData2)
# print(Annclf2.score(devData2,devDataCls2))
# plot_confusion(devDataCls2,devDataPred,"confMat2.jpg")


# def convert():
#     path = "TeluguLetters/"
#     folders = [path+"ai",path+"bA",path+"chA",path+"dA",path+"tA"]
#     min = 1e9 + 0.0
#     data_dict_train = {}
#     data_dict_dev = {}

#     for id,i in enumerate(folders):
#         data_dict_train[id]=[]
#         data_dict_dev[id]=[]
#         for j in os.listdir(i+"/train/"):
#             (filename,extension) = os.path.splitext(j)
#             if extension == '.txt':
#                 temp_file=open(i+"/train/"+j,"r")
#                 temp = []
#                 # flag = True
#                 for id2,k in enumerate(temp_file.readlines()):
#                     # print(k)
#                     # temp = k[:-2].split(" ")
#                     # print(temp)
#                     l = list(map(float,k[:-2].split(" ")))
#                     # if(flag):
#                     #     print(l)
#                     #     flag=False
#                     if(min>l[0]):
#                         min=l[0]
#                     for k in range(int(l[0])):
#                         temp.append([l[k*2+1],l[k*2+2]])
                    
#                 data_dict_train[id].append(temp)

#         for j in os.listdir(i+"/dev/"):
#             (filename,extension) = os.path.splitext(j)
#             if extension == '.txt':
#                 temp_file=open(i+"/dev/"+j,"r")
#                 temp = []
#                 for id2,k in enumerate(temp_file.readlines()):
#                     l = list(map(float,k[:-2].split(" ")))
#                     if(min>l[0]):
#                         min=l[0]
#                     for k in range(int(l[0])):
#                         temp.append([l[k*2+1],l[k*2+2]])
                    
#                 data_dict_dev[id].append(temp)
    
#     for i in data_dict_train.keys():
#         temp = []
#         for k in range(len(data_dict_train[i])):
#             scaler = MinMaxScaler()
#             scaler.fit(data_dict_train[i][k])
#             temp.append(scaler.transform(data_dict_train[i][k]))
#         data_dict_train[i]=temp
    
#     for i in data_dict_dev.keys():
#         temp = []
#         for k in range(len(data_dict_dev[i])):
#             scaler = MinMaxScaler()
#             scaler.fit(data_dict_dev[i][k])
#             temp.append(scaler.transform(data_dict_dev[i][k]))
#         data_dict_dev[i]=temp

#     min = int(min)

#     for i in data_dict_train.keys():
#         temp1 = []
#         for j in range(len(data_dict_train[i])):
#             temp2 = []
#             club = len(data_dict_train[i][j])-min+1
#             for k in range(min):
#                 avg = []
#                 for l in range(club):
#                     if(len(avg)==0):
#                         avg=data_dict_train[i][j][k+l]
#                     else:
#                         for m in range(len(avg)):
#                             avg[m]+=data_dict_train[i][j][k+l][m]

#                 for m in range(len(avg)):
#                     avg[m]=avg[m]/club
#                 temp2.extend(avg)
#             temp1.append(temp2)

#         data_dict_train[i] = temp1

#     for i in data_dict_dev.keys():
#         temp1 = []
#         for j in range(len(data_dict_dev[i])):
#             temp2 = []
#             club = len(data_dict_dev[i][j])-min+1
#             for k in range(min):
#                 avg = []
#                 for l in range(club):
#                     if(len(avg)==0):
#                         avg=data_dict_dev[i][j][k+l]
#                     else:
#                         for m in range(len(avg)):
#                             avg[m]+=data_dict_dev[i][j][k+l][m]

#                 for m in range(len(avg)):
#                     avg[m]=avg[m]/club
#                 temp2.extend(avg)
#             temp1.append(temp2)

#         data_dict_dev[i] = temp1
    
#     lst_train = []
#     lst_dev = []
#     map_train = []
#     map_dev = []
#     for i in data_dict_train.keys():
#         for ex in data_dict_train[i]:
#         	lst_train.append(ex)
#         	map_train.append(i)
#     for i in data_dict_dev.keys():
#     	for ex in data_dict_dev[i]:
#     		lst_dev.append(ex)
#     		map_dev.append(i)

#     return data_dict_train,data_dict_dev,lst_train,lst_dev,map_train,map_dev
# tmp,_,trainData3,devData3,trainDataCls3,devDataCls3 =  convert()
# trainData3,qs = PCA(trainData3,10) #change here for features
# #trainData3,trainDataCls3,qs3 = lda(trainData3,trainDataCls3)
# devData3 = reducedPCA(devData3,qs)
# #devData3 = reducedPCA(devData3,qs2)
# writeFile = open("ANN.txt",'w')
# clsCnt3 = len(tmp.keys())
# Annclf3 = MLPClassifier(hidden_layer_sizes=(100,33),activation='tanh',max_iter=1000).fit(trainData3,trainDataCls3)
# for i,d in enumerate(devData3):
# 	writeFile.write(str(devDataCls3[i]))
# 	writeFile.write(" ")
# 	probs = Annclf3.predict_proba([devData3[i]])[0]
# 	for j in range(clsCnt3):
# 		writeFile.write(str(probs[j]))
# 		writeFile.write(" ")
# 	writeFile.write("\n")
# writeFile.close()
# devDataPred = Annclf3.predict(devData3)
# print(Annclf3.score(devData3,devDataCls3))
# plot_confusion(devDataCls3,devDataPred,"confMat3.jpg")

tmp,_,trainData3,devData3,trainDataCls3,devDataCls3 =  spoken.convert()
#trainData3,qs = PCA(trainData3,10) #change here for features
#trainData3,trainDataCls3,qs3 = lda(trainData3,trainDataCls3)
#devData3 = reducedPCA(devData3,qs)
#devData3 = reducedPCA(devData3,qs2)
writeFile = open("ANN.txt",'w')
clsCnt3 = len(tmp.keys())
Annclf3 = MLPClassifier(random_state=10,hidden_layer_sizes=(80,16),activation='tanh',max_iter=1000).fit(trainData3,trainDataCls3)
for i,d in enumerate(devData3):
	writeFile.write(str(devDataCls3[i]))
	writeFile.write(" ")
	probs = Annclf3.predict_proba([devData3[i]])[0]
	for j in range(clsCnt3):
		writeFile.write(str(probs[j]))
		writeFile.write(" ")
	writeFile.write("\n")
writeFile.close()
devDataPred = Annclf3.predict(devData3)
print(Annclf3.score(devData3,devDataCls3))
plot_confusion(devDataCls3,devDataPred,"confMat3.jpg")