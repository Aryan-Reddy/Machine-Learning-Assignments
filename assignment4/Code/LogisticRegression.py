import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import os
import random
import seaborn as sb
import matplotlib.pylab as plt
import spoken
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
		qs.append(list(np.real(v[:,ind[j]])))
	for i in range(n):
		reducedData.append([])
		for j in range(ncomp):
			ind2 = ind[j]
			vec = v[:,ind2]
			reducedData[-1].append(np.real(np.dot(vec,data[i])))
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
def getYs(phiX,ws):
	clsCnt = len(ws)
	n = len(phiX)
	aas = []
	for w in ws:
		aas.append(np.dot(w,phiX))
	eaas = np.exp(aas)
	s = np.sum(eaas)
	ys = []
	for ea in eaas:
		ys.append(ea/s)
	return ys
def updates(phiXs,ts,wolds,eeta):
	n = len(phiXs)
	clsCnt = len(wolds)
	fCnt = len(phiXs[0])
	updates = np.zeros(wolds.shape)
	ys = []
	for i in range(n):
		ys.append(getYs(phiXs[i],wolds))
	for c in range(clsCnt):
		for j in range(fCnt):
			for i in range(n):
				updates[c][j] += (ys[i][c] - (int(c == ts[i])))*phiXs[i][j]
	return eeta * updates
def genPhiXs(basisLst,data):
	phiXs = []
	for e in data:
		phiXs.append([])
		for f in basisLst:
			phiXs[-1].append(f(e))
	return phiXs
# trainFile = open("SyntheticData/train.txt",'r')
# clsCnt = 2
# trainData = []
# trainDataCls = []
# writeFile = open("LR.txt",'w')
# Eeta = 10 ** (-4.5)
# for line in trainFile:
# 	words = line.split(',')
# 	n = len(words)
# 	c = (int(words[-1]) - 1)
# 	trainDataCls.append(c)
# 	trainData.append([])
# 	for i in range(n-1):
# 		trainData[-1].append(float(words[i]))
# #trainData,qs = PCA(trainData,1) #change here for feature
# #trainData,trainDataCls,qs2 = lda(trainData,trainDataCls)
# # x = [f[0] for f in trainData[0]]
# # y = [f[1] for f in trainData[0]]
# # plt.scatter(x,y,color='blue')
# # x = [f[0] for f in trainData[1]]
# # y = [f[1] for f in trainData[1]]
# # plt.scatter(x,y,color='red')
# # plt.savefig("tmp.jpg")
# #basisLst = [(lambda x: 1),(lambda x:x[0]),(lambda x:x[0] * x[0]/12),(lambda x:x[0] * x[0] * x[0]/144)]
# basisLst = [(lambda x: 1),(lambda x:x[0]),(lambda x:x[1]),(lambda x:x[0] * x[0]/12),(lambda x:x[0] * x[1]/12),(lambda x:x[1]*x[1]/12)]
# #print(len(basisLst))
# fCnt = len(basisLst)
# phiXs = genPhiXs(basisLst,trainData)
# clf = LogisticRegression(random_state=0,max_iter=500).fit(phiXs,trainDataCls)
# ws = np.zeros((clsCnt,fCnt))
# for it in range(300):
# 	tmp = updates(phiXs,trainDataCls,ws,Eeta)
# 	ws = ws - tmp
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
# devPhiXs = genPhiXs(basisLst,devData)
# devExCnt = len(devPhiXs)
# crct = 0
# totDev = 0
# crct2 = 0
# for i in range(devExCnt):
# 	writeFile.write(str(devDataCls[i])+" ")
# 	ys = getYs(devPhiXs[i],ws)
# 	for y in ys:
# 		writeFile.write(str(y) + " ")
# 	writeFile.write("\n")
# 	if ys.index(max(ys))==devDataCls[i]:
# 		crct += 1
# 	totDev += 1
# 	if clf.predict([devPhiXs[i]])[0] == devDataCls[i]:
# 		crct2 += 1
# print(100.0 * crct/totDev)
# print(crct,totDev)
# print(100.0 * crct2/totDev)
# print(crct2,totDev)


# imgTypes = ['coast','forest','highway','mountain','opencountry']
# writeFile = open("LR.txt",'w')
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
# trainData2 = list(norm2.transform(trainData2))
# trainData2,qs = PCA(trainData2,35) #change here for features
# # trainData2,trainDataCls2,qs2 = lda(trainData2,trainDataCls2)
# for t in trainData2:
# 	t = list(t)
# 	t.append(1)
# clf2 = LogisticRegression(random_state=0,max_iter=1000).fit(trainData2,trainDataCls2)
# fCnt2 = len(trainData2[0])
# Eeta2 = 10**-4.2
# ws = np.zeros((clsCnt2,fCnt2))
# for it in range(400):
# 	if it%100==0:
# 		print(it," done")
# 	tmp = updates(trainData2,trainDataCls2,ws,Eeta2)
# 	ws = ws - tmp
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
# devData2 = list(norm2.transform(devData2))
# devData2 = reducedPCA(devData2,qs)
# # devData2 = reducedPCA(devData2,qs2)
# for d in devData2:
# 	d = list(d)
# 	d.append(1)
# crct = 0
# crct2 = 0
# totDev = 0
# devExCnt2 = len(devData2)
# predCls2 = []
# predCls22 = []
# for i in range(devExCnt2):
# 	ys = getYs(devData2[i],ws)
# 	writeFile.write(str(devDataCls2[i]) + " ")
# 	for y in ys:
# 		writeFile.write(str(y) + " ")
# 	writeFile.write("\n")
# 	predCls2.append(ys.index(max(ys))) 
# 	if predCls2[-1]==devDataCls2[i]:
# 		crct += 1
# 	predCls22.append(clf2.predict([devData2[i]])[0])
# 	if  predCls22[-1] == devDataCls2[i]:
# 		crct2 += 1
# 	totDev += 1
# writeFile.close()
# print(100.0 * crct/totDev)
# print(100.0 * crct2/totDev)
# plot_confusion(predCls2,devDataCls2,"confMat2.jpg")
# plot_confusion(predCls22,devDataCls2,"confMat22.jpg")


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
# #trainData3,qs = PCA(trainData3,10) #change here for features
# #trainData3,trainDataCls3,qs2 = lda(trainData3,trainDataCls3)
# #devData3 = reducedPCA(devData3,qs)
# #devData3 = reducedPCA(devData3,qs2)
# writeFile = open("LR.txt",'w')
# clsCnt3 = len(tmp.keys())
# clf3 = LogisticRegression(random_state=0,max_iter=1000).fit(trainData3,trainDataCls3)
# for i,d in enumerate(devData3):
# 	writeFile.write(str(devDataCls3[i]))
# 	writeFile.write(" ")
# 	probs = clf3.predict_proba([devData3[i]])[0]
# 	for j in range(clsCnt3):
# 		writeFile.write(str(probs[j]))
# 		writeFile.write(" ")
# 	writeFile.write("\n")
# writeFile.close()
# devDataPred = clf3.predict(devData3)
# print(clf3.score(devData3,devDataCls3))
# plot_confusion(devDataCls3,devDataPred,"confMat3.jpg")


tmp,_,trainData3,devData3,trainDataCls3,devDataCls3 =  spoken.convert()
#trainData3,qs = PCA(trainData3,10) #change here for features
#trainData3,trainDataCls3,qs2 = lda(trainData3,trainDataCls3)
#devData3 = reducedPCA(devData3,qs)
#devData3 = reducedPCA(devData3,qs2)
writeFile = open("LR.txt",'w')
clsCnt3 = len(tmp.keys())
clf3 = LogisticRegression(random_state=0,max_iter=1000).fit(trainData3,trainDataCls3)
for i,d in enumerate(devData3):
	writeFile.write(str(devDataCls3[i]))
	writeFile.write(" ")
	probs = clf3.predict_proba([devData3[i]])[0]
	for j in range(clsCnt3):
		writeFile.write(str(probs[j]))
		writeFile.write(" ")
	writeFile.write("\n")
writeFile.close()
devDataPred = clf3.predict(devData3)
print(clf3.score(devData3,devDataCls3))
plot_confusion(devDataCls3,devDataPred,"confMat3.jpg")