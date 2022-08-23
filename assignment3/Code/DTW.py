import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay
INF = 1e18
topK = [1,3,5,10,15]
fig, [ax_roc,ax_det] = plt.subplots(1,2,figsize=(20,10))
def roc_plots(y,allclasses,ax,ax2):
	for k in range(len(topK)):
		allProbs = []
		for j in range(len(y)):
			for p in allclasses[j][k]:
				allProbs.append(p)
		th = np.sort(allProbs)
		tpr=[]
		fpr=[]
		fnr = []
		rates=[]
		for threshhold in th:
			(tp,fp,fn,tn)=(0.0,0.0,0.0,0.0)
			for i in range(len(y)):
				for j in range(len(allclasses[i][k])):
					if(allclasses[i][k][j]>=threshhold):#predict positive
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
		ax.plot(fpr,tpr,label="Case: k = " + str(topK[k]))
		DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="Case: k = " + str(topK[k])).plot(ax=ax2)
def plot_confusion_matrix(ll1,l2,class_count):
	# l1 is predicted and l2 is actual
	global topK
	for i in range(len(topK)):
		tmpStr = "confMatForK"+str(topK[i]) + ".jpg"
		l1 = [ll1[j][i] for j in range(len(l2))]
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
		fig.savefig(tmpStr)
def getDist(lstA,lstB):
	n = len(lstA)
	m = len(lstB)
	assert(n == m)
	dist = 0
	for i in range(n):
		diff = lstA[i] - lstB[i]
		dist += diff * diff
	return math.sqrt(dist)
def getDTWDist(framesA,framesB):
	global INF
	n = len(framesA)
	m = len(framesB)
	dp = np.zeros((n+1,m+1))
	for i in range(n+1):
		for j in range(m+1):
			dp[i][j] = INF
	dp[0][0] = 0
	for i in range(1,n+1):
		for j in range(1,m+1):
			cost = getDist(framesA[i-1],framesB[j-1])
			dp[i][j] = min(dp[i-1][j],min(dp[i-1][j-1],dp[i][j-1])) + cost
	return dp[n][m]
def predictClass(trainData,framesA):
	global topK
	print("REACHED")
	clsCnt = len(trainData)
	distances = []
	for i in range(clsCnt):
		distances.append([])
	avgDist = []
	for i in range(len(topK)):
		avgDist.append([])
	for k,clsLst in enumerate(trainData):
		for framesB in clsLst:
			distances[k].append((getDTWDist(framesA,framesB),k))
		distances[k].sort()
		for ind,tk in enumerate(topK):
			avgDist[ind].append(0.0)
			for i in range(tk):
				avgDist[ind][-1] +=distances[k][i][0]
			avgDist[ind][-1] /= tk
	#print(distances)
	confVals = []
	for i in range(len(topK)):
		confVals.append([])
		for d in avgDist[i]:
			confVals[-1].append(-d)
	pred = []
	for i in range(len(topK)):
		p = confVals[i].index(max(confVals[i]))
		pred.append(p)
	return (pred,confVals)
trainData = []
devData = []
digList = [1,3,4,7,9]
for num in digList:
	devDir = "Digits/" + str(num) + "/dev"
	trainDir = "Digits/" + str(num) + "/train"
	tlst = []
	dlst = []
	for Dir in [devDir,trainDir]:
		for file in os.listdir(Dir):
			filePath = os.path.join(Dir,file)
			if filePath.find(".mfcc") != -1 :
				fptr = open(filePath,'r')
				tmp = fptr.readline().split()
				(NC,NF) = (int(tmp[0]),int(tmp[1]))
				frames = []
				for line in fptr:
					tmpFrame = []
					for w in line.split():
						tmpFrame.append(float(w))
					frames.append(tmpFrame)
				if Dir == trainDir:
					tlst.append(frames)
				else:
					dlst.append(frames)
	trainData.append(tlst)
	devData.append(dlst)
predCls = []
actualCls = []
confVals = []
for k,clsLst in enumerate(devData):
	for framesA in clsLst:
		k2,conf = predictClass(trainData,framesA)
		predCls.append(k2)
		actualCls.append(k)
		confVals.append(conf)
		print("Class Predicted = ",k2,"Actual Class = ",k)
plot_confusion_matrix(predCls,actualCls,len(digList))
roc_plots(actualCls,confVals,ax_roc,ax_det)
ax_roc.legend()
fig.savefig("det_and_roc.jpg")