import os
import numpy as np
import math
import matplotlib.pyplot as plt
import shutil
from sklearn.cluster import KMeans
from sklearn.metrics import DetCurveDisplay
def minMaxNormalize(f):
	n = len(f)
	m = len(f[0])
	mnVal = [1e10] * n
	mxVal = [-1e10] * n
	for p in f:
		for i in range(m):
			mnVal[i] = min(mnVal[i],p[i])
			mxVal[i] = max(mxVal[i],p[i])
	tmp = []
	for p in f:
		tmp.append([])
		for i in range(m):
			tmp[-1].append((p[i] - mnVal[i])/(mxVal[i] - mnVal[i]))
	return tmp
def zscoreNormalize(f):
	return stats.zscore(f)
fig, [ax_roc,ax_det] = plt.subplots(1,2,figsize=(20,10))
def roc_plots(y,allclasses,ax,ax2,case):
	allProbs = []
	for lst in allclasses:
		for p in lst:
			allProbs.append(p)
	th = np.sort(allProbs)
	tpr=[]
	fpr=[]
	fnr = []
	rates=[]
	for threshhold in th:
		(tp,fp,fn,tn)=(0.0,0.0,0.0,0.0)
		for i in range(len(y)):
			for j in range(len(allclasses[i])):
				if(allclasses[i][j]>=threshhold):#predict positive
					if(y[i]==j):
						tp+=1;
					else:
						fp+=1
						
				else:
					if(y[i]==j):
						fn+=1
					else:
						tn+=1
		rates.append([tp/(tp+fn),fp/(fp+tn)])
		fnr.append(fn/(tp+fn))
	tpr=[i[0] for i in rates]
	fpr=[i[1] for i in rates]
	ax.plot(fpr,tpr,label="Case: " + str(case))
	DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="Case: " + case).plot(ax=ax2)
	return (fpr,fnr)
def plot_confusion_matrix(l1,l2,class_count,address):
	# l1 is predicted and l2 is actual
	fig,ax=plt.subplots(1,1)
	n=len(l1)
	m=np.zeros((class_count+1,class_count+1))
	for i in range(n):
		pred=l1[i]
		out=l2[i]
		m[out][pred]+=1
	correct=0
	total=0
	for i in range(class_count):
		row_correct=0
		row_total=0
		col_correct=0
		col_total=0
		for j in range(class_count):
			if(i==j):
				row_correct+=m[i][j]
				col_correct+=m[j][i]
			row_total+=m[i][j]
			col_total+=m[j][i]
		if(row_total!=0):
			m[i][-1]=(row_correct/row_total)*100
		if(col_total!=0):
			m[-1][i]=(col_correct/col_total)*100
		correct+=row_correct
		total+=row_total
			
	if(total!=0):
		m[-1][-1]=(correct/total)*100
	column_labels=[]
	for i in range(class_count):
		column_labels.append(str(i+1))
	column_labels.append("total % correct")
	ax.axis('tight')
	ax.axis('off')
	ax.table(cellText=m,rowLabels=column_labels,colLabels=column_labels,loc="center")
	fig.tight_layout()
	fig.savefig(address)
def getProbLeftRight(seq,reca,nxta,recaSymb,nxtaSymb):
	n = len(seq)
	statesCnt = len(reca)
	alpha = np.zeros(statesCnt)
	alpha[0] = 1
	for t in range(n):
		nalpha = np.zeros(statesCnt)
		for i in range(statesCnt):
			nalpha[i] += alpha[i] * reca[i] * recaSymb[i][seq[t]]
			if i > 0:
				nalpha[i] += alpha[i-1] * nxta[i-1] * nxtaSymb[i-1][seq[t]]
		alpha,nalpha = nalpha,alpha
	return np.sum(alpha)
for symbolCnt,statesCnt in [(15,3),(35,3),(30,3),(30,4)]:
	trainData = []
	devData = []
	letterList = ["ai","bA","chA","dA","tA"]
	for letter in letterList:
		devDir = "TeluguLetters/" + letter + "/dev"
		trainDir = "TeluguLetters/" + letter + "/train";
		tlst = []
		dlst = []
		for Dir in [devDir,trainDir]:
			for file in os.listdir(Dir):
				filePath = os.path.join(Dir,file)
				if filePath.find(".txt") != -1 :
					fptr = open(filePath,'r')
					tmp = fptr.readline().split()
					NF = int(tmp[0])
					frames = []
					for i in range(NF):
						x = float(tmp[2*i+1])
						y = float(tmp[2*i+2])
						frames.append([x,y])
					frames = minMaxNormalize(frames)
					if Dir == devDir:
						dlst.append(frames)
					else:
						tlst.append(frames)
		trainData.append(tlst)
		devData.append(dlst)
	confMat = np.zeros((len(letterList),len(letterList)))
	tmpdataLst = []
	for c,exampleLst in enumerate(trainData):
		for ind,example in enumerate(exampleLst):
			for fNum,frame in enumerate(example):
				tmpdataLst.append(frame)
	for c,exampleLst in enumerate(devData):
		for ind,example in enumerate(exampleLst):
			for fNum,frame in enumerate(example):
				tmpdataLst.append(frame)
	dataLst = np.array(tmpdataLst)
	kmeans = KMeans(n_clusters=symbolCnt,max_iter=50,random_state=100000).fit(dataLst)
	lab = kmeans.labels_
	trainData2 = []
	l = 0
	for c,exampleLst in enumerate(trainData):
		trainData2.append([])
		for ind,example in enumerate(exampleLst):
			trainData2[c].append([])
			for fNum,frame in enumerate(example):
				trainData2[c][ind].append(lab[l])
				l += 1
	HMMModels = []
	for c,exampleLst in enumerate(trainData2):
		file = open("testForTrain.hmm.seq","w")
		for ind,example in enumerate(exampleLst):
			for s in example:
				file.write(str(s) + " ")
			file.write("\n")
		tmpStr = "./train_hmm testForTrain.hmm.seq 100000 "+ str(statesCnt) + " " + str(symbolCnt) + " 0.001"
		#print(tmpStr)
		file.close()
		os.system(tmpStr)
		file = open("testForTrain.hmm.seq.hmm","r")
		file.readline()
		file.readline()
		reca = []
		nxta = []
		recaSymb = []
		nxtaSymb = []
		c = 0
		for line in file:
			words = line.split()
			if len(words) == 0:
				continue
			if c %2 == 0:
				reca.append(float(words[0]))
				recaSymb.append([])
				for i in range(1,symbolCnt+1):
					recaSymb[-1].append(float(words[i]))
			else:
				nxta.append(float(words[0]))
				nxtaSymb.append([])
				for i in range(1,symbolCnt+1):
					nxtaSymb[-1].append(float(words[i]))
			c += 1
		HMMModels.append((reca,nxta,recaSymb,nxtaSymb))
	predCls = []
	actualCls = []
	allclasses = []
	for c,exampleLst in enumerate(devData):
		for ind,example in enumerate(exampleLst):
			symbolSeq = []
			for fNum,frame in enumerate(example):
				symbolSeq.append(kmeans.predict([frame])[0])
			probs = []
			for c2,(reca,nxta,recaSymb,nxtaSymb) in enumerate(HMMModels):
				probs.append(getProbLeftRight(symbolSeq,reca,nxta,recaSymb,nxtaSymb))
			ii = probs.index(max(probs))
			confMat[c][ii] += 1
			predCls.append(ii)
			actualCls.append(c)
			allclasses.append(probs)
	roc_plots(actualCls,allclasses,ax_roc,ax_det,"States = " + str(statesCnt) + " sym = " + str(symbolCnt))
	plot_confusion_matrix(predCls,actualCls,len(letterList),"CMat2" + str(statesCnt) + " " + str(symbolCnt) + ".jpg")
	for lst in confMat:
		for cnt in lst:
			print(cnt,end=' ')
		print()
ax_roc.legend()
fig.savefig("roc_and_det.jpg")



