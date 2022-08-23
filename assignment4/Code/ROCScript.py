import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import DetCurveDisplay
fig, [ax_roc,ax_det] = plt.subplots(1,2,figsize=(20,10))
clsCnt = 5
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
Dir = "Scores"
for fileName in os.listdir(Dir):
	fileAdd = os.path.join(Dir,fileName)
	fptr = open(fileAdd,'r')
	acutalCls = []
	cscrs = []
	for line in fptr:
		words = line.split(" ")
		acutalCls.append(int(float(words[0])))
		tmp = []
		for i in range(1,clsCnt+1):
			tmp.append(float(words[i]))
		s = np.sum(tmp)
		if s ==0:
			print(tmp)
		for t in tmp:
			t /= s
		cscrs.append(tmp)
	roc_plots(acutalCls,cscrs,ax_roc,ax_det,fileName)
ax_roc.legend()
ax_det.legend()
fig.savefig("roc_and_det.jpg")
