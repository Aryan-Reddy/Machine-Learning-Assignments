import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import math
trainDataAddress = "train3.txt" # change these addresses for training data and development data
devDataAddress = "dev3.txt"
fig, [ax_roc,ax_det] = plt.subplots(1,2,figsize=(20,10))
#This function plots roc curves and returns fpr and fnr
def roc_plots(y,allclasses,ax,case):
	minn = min( i[j] for i in allclasses for j in range(3))
	maxx = max( i[j] for i in allclasses for j in range(3))
	th=np.linspace(minn,maxx,1000)
	tpr=[]
	fpr=[]
	fnr = []
	rates=[]
	for threshhold in th:
		(tp,fp,fn,tn)=(0.0,0.0,0.0,0.0)
		for i in range(len(y)):
			for j in range(3):
				if(allclasses[i][j]>=threshhold):#predict positive
					if(y[i]==j+1):
						tp+=1;
					else:
						fp+=1
						
				else:
					if(y[i]==j+1):
						fn+=1
					else:
						tn+=1
		rates.append([tp/(tp+fn),fp/(fp+tn)])
		fnr.append(fn/(tp+fn))
	tpr=[i[0] for i in rates]
	fpr=[i[1] for i in rates]
	ax.plot(fpr,tpr,label="Case: " + str(case))
	return (fpr,fnr)

#This function is used to plot decision surfaces
def plot_decision_surfaces(x1,x2,x3,x,y,address):
	fig,ax=plt.subplots(1,1)
	xx1=[]
	xx2=[]
	xx3=[]
	for i in range(len(x)):
		if(y[i]==1):
			xx1.append(x[i])
		elif(y[i]==2):
			xx2.append(x[i])
		else:
			xx3.append(x[i])
	ax.scatter([i[0] for i in xx1],[i[1] for i in xx1],color='blue')
	ax.scatter([i[0] for i in xx2],[i[1] for i in xx2],color='red')
	ax.scatter([i[0] for i in xx3],[i[1] for i in xx3],color='green')

	ax.scatter([i[0] for i in x1],[i[1] for i in x1],color='red',s=0.5)
	ax.scatter([i[0] for i in x2],[i[1] for i in x2],color='pink',s=0.5)
	ax.scatter([i[0] for i in x3],[i[1] for i in x3],color='yellow',s=0.5)
	ax.set_title("decision boundary and surface")
	fig.tight_layout()
	fig.savefig(address)

#This function is used to plot gaussian surfaces
def plot_guassian(mus,sigmas,address):
	fig, ax = plt.subplots(1,1,figsize=(10,10))
	ax = plt.axes(projection='3d') 
	fig2,ax2 = plt.subplots(1,1,figsize=(10,10))
	a=[]
	b=[]
	for i in mus:
		a.append(i[0][0])
		b.append(i[1][0])
	a.sort()
	b.sort()
	for i in clsses:
		i-=1
		w,v = np.linalg.eig(sigmas[i])
		distr=multivariate_normal(cov = sigmas[i], mean = [mus[i][0][0],mus[i][1][0]],seed=1000)
		x = np.linspace(a[0]-10,a[-1]+10 , num=100)
		y = np.linspace(b[0]-10,b[-1]+10, num=100)
		X, Y = np.meshgrid(x,y)
		pdf = np.zeros(X.shape)
		for k in range(X.shape[0]):
			for j in range(X.shape[1]):
				pdf[k,j]=distr.pdf([X[k,j], Y[k,j]])
		ax.contour3D(X,Y,pdf, 50, cmap='binary')
		ax2.contour(X,Y,pdf,5,cmap='binary')
		ax2.arrow(mus[i][0][0],mus[i][1][0],v[0][0]*5,v[1][0]*5)
		ax2.arrow(mus[i][0][0],mus[i][1][0],v[0][1]*5,v[1][1]*5)
	ax.view_init(30,0)
	fig.savefig(address)
	fig2.savefig("contour" + address)
	plt.close()
	#plt.figure(fig.number)
	#plt.close()
#This function is used to plot confusion matrices
def plot_confusion_matrix(l1,l2,class_count,address):
	# l1 is predicted and l2 is actual
	fig,ax=plt.subplots(1,1)
	n=len(l1)
	m=np.zeros((class_count+1,class_count+1))
	for i in range(n):
		pred=l1[i]
		out=l2[i]
		m[out-1][pred-1]+=1
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
#This function is used to get p(ci) and means from data
def getPisMus(clsDataLst,featureCnt):
	pis = []
	tot = 0
	for lst in clsDataLst:
		tot += len(lst)
	for lst in clsDataLst:
		pis.append(float(len(lst))/tot)
	#print(pis)
	mus = []
	for lst in clsDataLst:
		mu = np.zeros((featureCnt,1))
		for featureLst in lst:
			for j in range(featureCnt):
				mu[j] += featureLst[j]
		mu /= len(lst)
		mus.append(mu)
	return (pis,mus)
#This function trains for case 1
def train1(clsDataLst,featureCnt):
	clsCnt = len(clsDataLst)
	(pis,mus) = getPisMus(clsDataLst,featureCnt)
	sigma = np.zeros((featureCnt,featureCnt))
	tot = 0
	for i,lst in enumerate(clsDataLst):
		tot += len(lst)
		for featureLst in lst:
			xarr = np.zeros((featureCnt,1))
			for k in range(featureCnt):
				xarr[k][0] = featureLst[k];
			diff = xarr - mus[i]
			diffT = diff.transpose()
			sigma += diff @ diffT
	sigma /= tot - clsCnt
	return (pis,mus,sigma)
#This function trains for case 2
def train2(clsDataLst,featureCnt):
	clsCnt = len(clsDataLst)
	(pis,mus) = getPisMus(clsDataLst,featureCnt)
	sigmas = []
	for i,lst in enumerate(clsDataLst):
		sigma = np.zeros((featureCnt,featureCnt))
		for featureLst in lst:
			xarr = np.zeros((featureCnt,1))
			for k in range(featureCnt):
				xarr[k][0] = featureLst[k];
			diff = xarr - mus[i]
			diffT = diff.transpose()
			sigma += diff @ diffT
		sigma /= len(lst) - 1
		sigmas.append(sigma)
	return (pis,mus,sigmas)
#This function predicts for case 2
def predictor1(pis,mus,sigmainv,x):
	clsCnt = len(pis)
	featureCnt = len(x)
	conVals = []
	xarr = np.zeros((featureCnt,1))
	for k in range(featureCnt):
		xarr[k][0] = x[k]
	dis = []
	for i in range(clsCnt):
		discrim = xarr.transpose() @ sigmainv @ mus[i] - 0.5 * (mus[i].transpose()) @ sigmainv @ mus[i] + np.log(pis[i])
		dis.append(discrim[0][0])
		conVals.append(np.exp(discrim[0][0]))
	return dis.index(max(dis)),conVals
#This function predicts for case 2
def predictor2(pis,mus,sigmaDets,sigmaInvs,x):
	clsCnt = len(pis)
	dis = []
	xarr = np.zeros((len(x),1))
	for i,x in enumerate(x):
		xarr[i][0] = x
	conVals = []
	for i in range(clsCnt):
		diff = (xarr - mus[i])
		diffT = diff.transpose()
		discrim = np.log(pis[i]) - 0.5 * np.log(sigmaDets[i]) - 0.5 * diffT @ sigmaInvs[i] @ diff
		conVals.append(np.exp(discrim[0]))
		dis.append(discrim)
	return (dis.index(max(dis)),conVals)
#This function trains for case 3
def train3(clsDataLst,featureCnt):
	clsCnt = len(clsDataLst)
	tot = 0
	(pis,mus) = getPisMus(clsDataLst,featureCnt)
	sigma = 0.0
	for (i,lst) in enumerate(clsDataLst):
		tot += len(lst)
		for featureLst in lst:
			for k in range(featureCnt):
				diff = (featureLst[k] - mus[i][k])
				sigma += diff * diff
	sigma /= tot * featureCnt - clsCnt * featureCnt
	return (pis,mus,sigma)
#This function predicts for case 3
def predictor3(pis,mus,sigma,x):
	clsCnt = len(pis)
	dis = []
	conVals = []
	for i in range(clsCnt):
		discrim = np.log(pis[i])
		for j in range(featureCnt):
			diff = x[j] - mus[i][j]
			discrim -= (diff * diff)/(2 * sigma)
		dis.append(discrim)
		conVals.append(np.exp(discrim[0]))
	return dis.index(max(dis)),conVals
#This function trains for case 4
def train4(clsDataLst,featureCnt):
	clsCnt = len(clsDataLst)
	tot = 0
	(pis,mus) = getPisMus(clsDataLst,featureCnt)
	sigma = np.zeros(featureCnt) #sigma[i] = variance of ith feature
	for (i,lst) in enumerate(clsDataLst):
		tot += len(lst)
		for featureLst in lst:
			for k in range(featureCnt):
				diff = (featureLst[k] - mus[i][k])
				sigma[k] += diff * diff
	for j in range(featureCnt):
		sigma[j] /= tot - clsCnt
	return (pis,mus,sigma)
#This function predicts for case 4
def predictor4(pis,mus,sigma,x):
	clsCnt = len(pis)
	dis = []
	conVals = []
	for i in range(clsCnt):
		discrim = np.log(pis[i])
		for j in range(featureCnt):
			diff = x[j] - mus[i][j]
			discrim -= (diff * diff)/(2 * sigma[j])
		dis.append(discrim)
		conVals.append(np.exp(discrim[0]))
	return dis.index(max(dis)),conVals
#This function trains for case 5
def train5(clsDataLst,featureCnt):
	clsCnt = len(clsDataLst)
	(pis,mus) = getPisMus(clsDataLst,featureCnt)
	sigma = np.zeros((clsCnt,featureCnt)) #sigma[i][j] is variance of jth feature of ith class
	for (i,lst) in enumerate(clsDataLst):
		for featureLst in lst:
			for k in range(featureCnt):
				diff = (featureLst[k] - mus[i][k])
				sigma[i][k] += diff * diff
	for i in range(clsCnt):
		for j in range(featureCnt):
			sigma[i][j] /= len(clsDataLst[i]) - 1
	return (pis,mus,sigma)
#This function predicts for case 5
def predictor5(pis,mus,sigma,x):
	clsCnt = len(pis)
	dis = []
	conVals = []
	for i in range(clsCnt):
		discrim = np.log(pis[i])
		for j in range(featureCnt):
			diff = x[j] - mus[i][j]
			discrim -= 0.5 * np.log(sigma[i][j]) + (diff * diff)/(2 * sigma[i][j])
		dis.append(discrim)
		conVals.append(np.exp(discrim[0]))
	return dis.index(max(dis)),conVals
clsDataDict = {}
trainData = []
trianLabels = []
fInput = open(trainDataAddress,'r')
featureCnt = -1
for line in fInput:
	lstI = line.split(sep=',')
	n = len(lstI)
	c = int(lstI[n-1])
	features = []
	for i in range(n-1):
		features.append(float(lstI[i]))
	trainData.append(features)
	trianLabels.append(c)
	if c in clsDataDict:
		clsDataDict[c].append(features)
	else:
		clsDataDict[c] = [features]
	featureCnt = n-1
clsDataLst = []
clsses = []
for (key,lst) in clsDataDict.items():
	clsses.append(key)
	clsDataLst.append(lst)
# #case 1
# Train and predict data assuming case 1 model
def case1Scope():
	fDev = open(devDataAddress,'r')
	(pis,mus,sigma) = train1(clsDataLst,featureCnt)
	plot_guassian(mus,[sigma,sigma,sigma],"Case1Plot.jpg")
	sigmaInv = np.linalg.inv(sigma)
	predCls = []
	actualCls = []
	conVals = []
	tot = 0
	correct = 0
	mn = [1e9,1e9]
	mx = [-1e9,-1e9]
	for line in fDev:
		lstI = line.split(sep=',')
		n = len(lstI)
		c = int(lstI[n-1])
		features = []
		for i in range(n-1):
			features.append(float(lstI[i]))
			mn[i] = min(mn[i],features[-1])
			mx[i] = max(mx[i],features[-1])
		ind,tmpConVals = predictor1(pis,mus,sigmaInv,features)
		conVals.append(tmpConVals)
		predCls.append(clsses[ind])
		actualCls.append(c)
		if clsses[ind] == c:
			correct += 1
		tot += 1
	print(correct,tot)
	(fpr,fnr)=roc_plots(actualCls,conVals,ax_roc,1)
	display = DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="Case: 1").plot(ax=ax_det)
	plot_confusion_matrix(predCls,actualCls,3,"ConfusionCase1,jpg")
	fDev.close()
	mn1 = mn[0]
	mn2 = mn[1]
	mx1 = mx[0]
	mx2 = mx[1]
	itr1=mn1-10
	mg=[]
	add1=(mx1-mn1)/100
	add2=(mx2-mn2)/100
	predy=[]
	while(itr1<=mx1+10):
		itr2=mn2-10
		while(itr2<=mx2+10):
			mg.append([itr1,itr2])
			predy.append(predictor1(pis,mus,sigmaInv,[itr1,itr2])[0])
			itr2+=add2
		itr1+=add1
			

	plot_decision_surfaces(clsDataDict[1],clsDataDict[2],clsDataDict[3],mg,predy,"boundaries1.jpg")
# Train and predict data assuming case 2 model
def case2Scope():
	fDev = open(devDataAddress,'r')
	(pis,mus,sigmas) = train2(clsDataLst,featureCnt)
	plot_guassian(mus,sigmas,"Case2Plot.jpg")
	sigmaDets = []
	sigmaInvs = []
	predCls = []
	actualCls = []
	for sigma in sigmas:
		sigmaDets.append(np.linalg.det(sigma))
		sigmaInvs.append(np.linalg.inv(sigma))
	tot = 0
	correct = 0
	conVals = []
	mn = [1e9,1e9]
	mx = [-1e9,-1e9]
	for line in fDev:
		lstI = line.split(sep=',')
		n = len(lstI)
		c = int(lstI[n-1])
		features = []
		for i in range(n-1):
			features.append(float(lstI[i]))
			mn[i] = min(mn[i],features[-1])
			mx[i] = max(mx[i],features[-1])
		ind,tmpConVals = predictor2(pis,mus,sigmaDets,sigmaInvs,features)
		conVals.append(tmpConVals)
		predCls.append(clsses[ind])
		actualCls.append(c)
		if clsses[ind] == c:
			correct += 1
		tot += 1
	print(correct,tot)
	(fpr,fnr)=roc_plots(actualCls,conVals,ax_roc,2)
	display = DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="Case: 2").plot(ax=ax_det)
	plot_confusion_matrix(predCls,actualCls,3,"ConfusionCase2.jpg")
	fDev.close()
	mn1 = mn[0]
	mn2 = mn[1]
	mx1 = mx[0]
	mx2 = mx[1]
	itr1=mn1-10
	mg=[]
	add1=(mx1-mn1)/100
	add2=(mx2-mn2)/100
	predy=[]
	while(itr1<=mx1+10):
		itr2=mn2-10
		while(itr2<=mx2+10):
			mg.append([itr1,itr2])
			predy.append(predictor2(pis,mus,sigmaDets,sigmaInvs,[itr1,itr2])[0])
			itr2+=add2
		itr1+=add1
			

	plot_decision_surfaces(clsDataDict[1],clsDataDict[2],clsDataDict[3],mg,predy,"boundaries2.jpg")
# #case 3 & 4 & 5
# Train and predict data assuming case 3 4 and 5 models
def case345Scope():
	trn = {}
	prdF = {}
	trn[3] = train3
	trn[4] = train4
	trn[5] = train5
	prdF[3] = predictor3
	prdF[4] = predictor4
	prdF[5] = predictor5
	for case in [3,4,5]:
		fDev = open(devDataAddress,'r')
		(pis,mus,sigma) = trn[case](clsDataLst,featureCnt)
		if case == 3:
			tmp = np.diag(np.asarray([sigma[0] for i in range(featureCnt)]))
			plot_guassian(mus,[tmp,tmp,tmp],"Case3Plot.jpg")
		elif case == 4:
			tmp = np.diag(sigma)
			plot_guassian(mus,[tmp,tmp,tmp],"Case4Plot.jpg")
		else:
			plot_guassian(mus,[np.diag(sigma) for sigma in sigma],"Case5Plot.jpg")
		tot = 0
		correct = 0
		testData = []
		actualCls = []
		conVals = []
		predCls = []
		mn = [1e9,1e9]
		mx = [-1e9,-1e9]
		for line in fDev:
			lstI = line.split(sep=',')
			n = len(lstI)
			c = int(lstI[n-1])
			features = []
			for i in range(n-1):
				features.append(float(lstI[i]))
				mn[i] = min(mn[i],features[-1])
				mx[i] = max(mx[i],features[-1])
			testData.append(features)
			actualCls.append(c)
			ind,tmpConVals = prdF[case](pis,mus,sigma,features)
			conVals.append(tmpConVals)
			predCls.append(clsses[ind])
			if clsses[ind] == c:
				correct +=1
			tot += 1
		print(correct,tot)
		(fpr,fnr)=roc_plots(actualCls,conVals,ax_roc,case)
		display = DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="Case: " + str(case)).plot(ax=ax_det)
		plot_confusion_matrix(predCls,actualCls,3,"ConfusionCase" + str(case) + ".jpg")
		fDev.close()
		mn1 = mn[0]
		mn2 = mn[1]
		mx1 = mx[0]
		mx2 = mx[1]
		itr1=mn1-10
		mg=[]
		add1=(mx1-mn1)/100
		add2=(mx2-mn2)/100
		predy=[]
		while(itr1<=mx1+10):
			itr2=mn2-10
			while(itr2<=mx2+10):
				mg.append([itr1,itr2])
				predy.append(prdF[case](pis,mus,sigma,[itr1,itr2])[0])
				itr2+=add2
			itr1+=add1
				

		plot_decision_surfaces(clsDataDict[1],clsDataDict[2],clsDataDict[3],mg,predy,"boundaries" + str(case) + ".jpg")
#calling the function to train models
case1Scope()
case2Scope()
case345Scope()
ax_roc.legend()
#save roc and det curves
fig.savefig("det_and_roc.jpg")

