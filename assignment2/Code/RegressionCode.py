from re import A
# from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil
import math
from mpl_toolkits import mplot3d
FIG_CNT = 1
def power(x,n):
	cur = 1.0
	mult = x
	for i in range(n):
		cur *= mult
	return cur
def lse(phi,t,w):
	tmp = (t - phi @ w)
	return (np.transpose(tmp) @ tmp)
def y_value(x,w,bf):
	m = len(bf)
	tmp = np.zeros(m)
	for i in range(m):
		tmp[i] = bf[i](x)
	return tmp @ w
def ERMS(w,lstx,lsty,bf):
	n = len(lstx)
	error = 0.0
	for i in range(n):
		diff = y_value(lstx[i],w,bf) - lsty[i]
		error += diff * diff
	return math.sqrt(error/n)
def getPhi(featurelst,basisLst):
	n = len(featurelst)
	m = len(basisLst)
	phi = np.zeros((n,m))
	for i in range(n):
		for j in range(m):
			phi[i][j] = basisLst[j](featurelst[i])
	return phi
def train(phi,tarr,l):
	phiT = np.transpose(phi)
	pTp = phiT @ phi
	n = len(pTp)
	for i in range(n):
		pTp[i][i] += l
	return np.linalg.inv(pTp) @ phiT @ tarr
def plotAndSave(lstX,lstY,address,scatter=False):
	global FIG_CNT
	plt.figure(FIG_CNT)
	FIG_CNT += 1
	if not scatter:
		plt.plot(lstX,lstY)
	else:
		plt.scatter(lstX,lstY)
	if os.path.exists(address):
		os.remove(address)
	plt.savefig(address)
	plt.close()
def plotAndSave2(lstX,lstY,address,lstX2,lstY2,lstX3,lstY3,lstxtrain,lstxdev,scatter=False,is2d=False):
	# global FIG_CNT
	# plt.figure(FIG_CNT)
	# FIG_CNT += 1
	fig=plt.figure()
	if not scatter:
		ax=fig.add_subplot(2,2,1)
		ax.plot(lstX,lstY)
		ax=fig.add_subplot(2,2,2)
		ax.plot(lstX2,lstY2)
	else :
		ax=fig.add_subplot(2,2,1)
		ax.scatter(lstX,lstY,s=1)
		ax=fig.add_subplot(2,2,2)
		ax.scatter(lstX2,lstY2,s=1)

	
	if (not is2d):
		ax=fig.add_subplot(2,2,3)
		ax.plot(lstX3,lstY3)
		ax.scatter(lstxtrain,lstX,s=1)
		ax=fig.add_subplot(2,2,4)
		ax.plot(lstX3,lstY3)
		ax.scatter(lstxdev,lstX2,s=1)
	else:
		ax=fig.add_subplot(2,2,3,projection ='3d')
		ax.plot3D(lstX3[0],lstX3[1],lstY3,linewidth=0.05,color="r")
		ax.scatter(lstxtrain[0],lstxtrain[1],lstX,s=1)
		ax=fig.add_subplot(2,2,4,projection ='3d')
		ax.plot3D(lstX3[0],lstX3[1],lstY3,linewidth=0.05,color="r")
		ax.scatter(lstxdev[0],lstxdev[1],lstX2,s=1)

	if os.path.exists(address):
		os.remove(address)
	fig.savefig(address)
	plt.close(fig)
#Plotting function used for linear regression
def scatterPlotLinearReg(w,lstX,lstY,basisLst,lstxdev,lstydev,address,is2d=False):
	n = len(lstY)
	predT = []
	for i in range(n):
		predT.append(y_value(lstX[i],w,basisLst))
	m = len(lstydev)
	predT2 = []
	for i in range(m):
		predT2.append(y_value(lstxdev[i],w,basisLst))

	lstX3=[]
	lstY3=[]
	sendX = []
	sendXdev = []
	if (not is2d):
		i = 0
		while i <= 5.0:
			lstX3.append(i)
			lstY3.append(y_value([i],w,basisLst))
			i += 0.01
		
		for i in range(n):
			sendX.append(lstX[i][0])
			sendXdev.append(lstxdev[i][0])
	else:
		lstX3=[[],[]]
		i=-1
		while (i<=1):
			j=-1
			while(j<=1):
				lstX3[0].append(i)
				lstX3[1].append(j)
				lstY3.append(y_value([i,j],w,basisLst))
				j+=0.01
			i+=0.01
		sendX=[[],[]]
		sendXdev=[[],[]]
		for i in range(n):
			sendX[0].append(lstX[i][0])
			sendX[1].append(lstX[i][1])
			sendXdev[0].append(lstxdev[i][0])
			sendXdev[1].append(lstxdev[i][1])
	plotAndSave2(lstY,predT,address,lstydev,predT2,lstX3,lstY3,sendX,sendXdev,True,is2d)


storeERMS = {}
storeERMS2 = {}
storeERMS2d={}
storeERMS2d2={}

#This function takes train data, degree, lamda value and development data
#It then trains the model and predicts it on development data
def experiment(indices,N,deg,lam,x,y,indices2,xdev,ydev):
	lstx = []
	lsty = []
	lstxdev = []
	lstydev = []
	for i in range(N):
		lstx.append(x[indices[i]])
		lsty.append(y[indices[i]])
		lstxdev.append(xdev[indices2[i]])
		lstydev.append(ydev[indices2[i]])
	basislist=[]
	for i in range(deg+1):
		basislist.append(lambda z,i=i: power(z[0],i))
	tarr = np.asarray(lsty).transpose()
	phi = getPhi(lstx,basislist)
	w = train(phi,tarr,lam)
	imgAdd = ("N" + str(N) + "/D" + str(deg) + "/lam" + str(lam) + ".jpg")
	fileAdd = "N" + str(N) + "/D" + str(deg) + "/lam" + str(lam) + " wvals" + ".txt"
	scatterPlotLinearReg(w,lstx,lsty,basislist,lstxdev,lstydev,imgAdd)
	storeERMS[(N,deg,lam)] = ERMS(w,lstx,lsty,basislist)
	storeERMS2[(N,deg,lam)] = ERMS(w,lstxdev,lstydev,basislist)
	fFile = open(fileAdd,'w')
	for w in w:
		fFile.write(str(w)+'\n')
	fFile.close()
	return w
#This is same as previous function except for 2d features
def experiment2d(indices,N,deg,lam,x,y,indices2,xdev,ydev):
	lstx = []
	lsty = []
	lstxdev = []
	lstydev = []
	for i in range(N):
		lstx.append(x[indices[i]])
		lsty.append(y[indices[i]])
		lstxdev.append(xdev[indices2[i]])
		lstydev.append(ydev[indices2[i]])
	
	basislst=[]

	for i in range(deg+1):
		for j in range(i+1):
			basislst.append(lambda x,i=i,j=j: power(x[0],j)*power(x[1],i-j))
			
	tarr = np.asarray(lsty).transpose()
	phi = getPhi(lstx,basislst)
	w = train(phi,tarr,lam)
	imgAdd = ("2d-N" + str(N) + "/D" + str(deg) + "/lam" + str(lam) + ".jpg")
	fileAdd = "2d-N" + str(N) + "/D" + str(deg) + "/lam" + str(lam) + " wvals" + ".txt"
	scatterPlotLinearReg(w,lstx,lsty,basislst,lstxdev,lstydev,imgAdd,True)
	storeERMS2d[(N,deg,lam)] = ERMS(w,lstx,lsty,basislst)
	storeERMS2d2[(N,deg,lam)] = ERMS(w,lstxdev,lstydev,basislst)
	fFile = open(fileAdd,'w')
	for w in w:
		fFile.write(str(w)+'\n')
	fFile.close()
	return w
#Train for 1d data
fTrain = open("1d_team_1_train.txt",'r')
fDev = open("1d_team_1_dev.txt",'r')
lstX = []
lstY = []
lstXDEV = []
lstYDEV = []
for line in fTrain:
	cLst = line.split()
	n = len(cLst)
	tmpX = []
	for i in range(n-1):
		tmpX.append(float(cLst[i]))
	lstX.append(tmpX)
	lstY.append(float(cLst[n-1]))
for line in fDev:
	cLst = line.split()
	n = len(cLst)
	tmpX = []
	for i in range(n-1):
		tmpX.append(float(cLst[i]))
	lstXDEV.append(tmpX)
	lstYDEV.append(float(cLst[n-1]))
n = len(lstX)
n2 = len(lstXDEV)
fileRMS = open("RMSVALUES.txt",'w')
for N in []:
	print("PROGRESS N = " + str(N))
	folderAddress = "N" + str(N)
	if os.path.exists(folderAddress):
		shutil.rmtree(folderAddress)
	os.mkdir(folderAddress)
	ind1 = list(range(n))
	ind2 = list(range(n2))
	random.shuffle(ind1)
	random.shuffle(ind2)
	for deg in [8]:
		print("PROGRESS D = " + str(deg))
		os.mkdir(folderAddress + "/" + "D" + str(deg))
		for lam in [0,0.01,0.001,0.1,1,10]:
			print("LAM = " + str(lam))
			w = experiment(ind1,N,deg,lam,lstX,lstY,ind2,lstXDEV,lstYDEV)
			tup = (N,deg,lam)
			fileRMS.write(str(N) + ' ' + str(deg) + ' ' + str(lam) + ' ' + str(storeERMS[tup]) + ' ' + str(storeERMS2[tup]) + '\n')
fileRMS.close()
fTrain.close()

#Train for 2d data.
train2d=open("2d_team_1_train.txt","r")
dev2d=open("2d_team_1_dev.txt","r")

xtrain=[]
ttrain=[]

xdev=[]
tdev=[]

for i in train2d.readlines():
	xtrain.append([float( i[:-1].split(" ")[0]),float(i[:-1].split(" ")[1])])
	ttrain.append(float( i[:-1].split(" ")[2]))

for i in dev2d.readlines():
	xdev.append([float( i[:-1].split(" ")[0]),float(i[:-1].split(" ")[1])])
	tdev.append(float( i[:-1].split(" ")[2]))
fileRMS2d = open("RMSVALUES2d.txt",'w')
n=len(ttrain)
n2=len(tdev)
for N in [1000]:
	print("PROGRESS N = " + str(N))
	folderAddress = "2d-N" + str(N)
	if os.path.exists(folderAddress):
		shutil.rmtree(folderAddress)
	os.mkdir(folderAddress)
	ind1 = list(range(n))
	ind2 = list(range(n2))
	random.shuffle(ind1)
	random.shuffle(ind2)
	for deg in [4]:
		print("PROGRESS D = " + str(deg))
		os.mkdir(folderAddress + "/" + "D" + str(deg))
		for lam in [0]:
			print("LAM = " + str(lam))
			w = experiment2d(ind1,N,deg,lam,xtrain,ttrain,ind2,xdev,tdev)
			tup = (N,deg,lam)
			fileRMS2d.write(str(N) + ' ' + str(deg) + ' ' + str(lam) + ' ' + str(storeERMS2d[tup]) + ' ' + str(storeERMS2d2[tup]) + '\n')
fileRMS2d.close()
train2d.close()
dev2d.close()