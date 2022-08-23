from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d
from sklearn.metrics import DetCurveDisplay
import math
import random
# random.seed(9233)
# function to plot the decision boundaries 
def decision_boundary(mus,covs,pis,x_dict,ax,name):
	# fig,ax=plt.subplots(1,1,figsize=(20,20))
	points=[i[0] for i in x_dict[1]]
	points.extend([i[0] for i in x_dict[2]])
	points.sort()
	X=np.linspace(points[0]-2,points[-1]+2,100)
	points=[i[1] for i in x_dict[1]]
	points.extend([i[1] for i in x_dict[2]])
	points.sort()
	Y=np.linspace(points[0]-2,points[-1]+2,100)
	a=[]
	b=[]
	c=[]
	d=[]
	for x in X:
		for y in Y:
			max_prob=0
			for j in range(len(mus)):
				temp_prob=0
				for k in range(len(mus[j])):
					# distr=multivariate_normal(mean=mus[j][k],cov=covs[j][k])
					temp_prob+=pis[j][k]*guassian(mus[j][k],covs[j][k],[x,y])
					if(temp_prob>max_prob):
						max_prob=temp_prob
						curr_class=j+1
					if(curr_class==1):
						a.append(x)
						b.append(y)
					else:
						c.append(x)
						d.append(y)
	ax.scatter(a,b,s=50,color='pink')
	ax.scatter(c,d,s=50,color='yellow')
	ax.scatter([i[0] for i in x_dict[1]] , [j[1] for j in x_dict[1]],s=10,color='blue')
	ax.scatter([i[0] for i in x_dict[2]] , [j[1] for j in x_dict[2]],s=10,color='red')
	fig.savefig(name+"decision_boundary_diagonal.jpg")
	plt.close()

# function to get accuracy for the synthetic data
def accuracy(x_vals,y_vals,mus,covs,pis):
	correct=0
	for idx,i in enumerate(x_vals):
		max_prob=0
		curr_class=0
		for j in range(len(mus)):
			temp_prob=0
			for k in range(len(mus[j])):
				# distr=multivariate_normal(mean=mus[j][k],cov=covs[j][k])
				temp_prob+=pis[j][k]*guassian(mus[j][k],covs[j][k],i)
			if(temp_prob>max_prob):
				max_prob=temp_prob
				curr_class=j+1
		if(curr_class==y_vals[idx]):
			correct+=1
	for i in range(len(mus)):
		print("k"+str(i)+" = "+str(len(mus[i])),end="  ")
	print("accuracy = " + str(100*correct/len(x_vals)))

# to get accuracy for the image data
def accuracy2(x_vals,y_vals,mus,covs,pis):
	correct=0
	temp=[0,0,0,0,0]
	temp2=[0,0,0,0,0]
	for idx,i in enumerate(x_vals):
		max_prob=0
		curr_class=0
		for j in range(len(mus)):
			sum=0.0
			for l in range(len(i)):
				temp_prob=0.0
				for k in range(len(mus[j])):
					pis[j][k]*guassian(mus[j][k],covs[j][k],x_vals[idx][l])
					temp_prob+=pis[j][k]*guassian(mus[j][k],covs[j][k],x_vals[idx][l])
				print(temp_prob)
				temp_prob+=1e-323
				sum+=math.log(temp_prob)
			if(sum>max_prob):
				max_prob=temp_prob
				curr_class=j+1
		if(curr_class==y_vals[idx]):
			correct+=1
			temp[curr_class-1]+=1
		temp2[y_vals[idx]-1]+=1
	for i in range(len(mus)):
		print("k"+str(i)+" = "+str(len(mus[i])),end="  ")
	print("accuracy = " + str(100*correct/len(x_vals)))
	print(temp)
	print(temp2)

# to get fpr and fnr 
def roc_plots(y,allclasses,ax,case,class_count):
	th=[]
	for i in allclasses:
		for j in i:
			th.append(j)
	th.sort()
	# minn = min( i[j] for i in allclasses for j in range(class_count))
	# maxx = max( i[j] for i in allclasses for j in range(class_count))
	# th=np.linspace(minn,maxx,1000)
	tpr=[]
	fpr=[]
	fnr = []
	rates=[]
	for threshhold in th:
		(tp,fp,fn,tn)=(0.0,0.0,0.0,0.0)
		for i in range(len(y)):
			for j in range(class_count):
				if(allclasses[i][j]>=threshhold):#predict positive
					if(y[i]==j+1):
						tp+=1
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

# to get euclidian distance 
def euclid_dist(a,b):
	ans=0
	for i in range(len(a)):
		ans+=pow((a[i]-b[i]),2);
	return pow(ans,1/2)

# to get arithmetic mean 
def arithmetic_mean(x,gammas,Nk):
	ans = []
	for i in x[0]:
		ans.append(0)
	for k,i in enumerate(x):
		for j in range(len(i)):
			ans[j]+=i[j]*gammas[k]
	for i in range(len(x[0])):
		ans[i]/=Nk
	return ans

# to get the covariance matrix 
def covariance(xs,mu,gammas,Nk):
	n = len(xs[0])
	cov=[]
	for i in range(n):
		temp=[]
		for j in range(n):
			temp.append(0)
		cov.append(temp)
	for i in range(n):
		for j in range(n):
			temp=0
			for k,x in enumerate(xs):
				temp+=(x[i]-mu[i])*(x[j]-mu[j])*gammas[k]
			temp/=Nk
			cov[i][j]=temp
	return cov

# to get the guassian value
def guassian(mu,cov,x,flag=False):
	# if(flag):
	# 	print(mu)
	# 	print(cov)
	# 	print(x)
	mu=np.array(mu)
	x=np.array(x)
	a=mu-x
	a=np.reshape(a,(np.size(mu),1))
	cov=np.array(cov)
	# if flag:
		# print(math.exp(-1*((a.T@(np.linalg.inv(cov))@a)[0][0]))/math.sqrt(np.linalg.det(cov)))
	return math.exp(-1*((a.T@(np.linalg.inv(cov))@a)[0][0]))/math.sqrt(np.linalg.det(cov))


# to perform k means algorithm
def k_means(xs,k,iter_count,class_no):
	mus=[]
	random_indices = random.sample([i for i in range(len(xs))],k)
	for i in random_indices:
		mus.append(xs[i])
	cluster_dict={}
	gammas=[]
	for i in range(len(xs)):
		gammas.append(1)
	for i in range(iter_count):
		# expectation
		cluster_dict={}
		for x in xs :
			min = euclid_dist(x,mus[0])
			cluster = 0
			for (j,mu) in enumerate(mus):
				if(min>euclid_dist(x,mu)):
					min=euclid_dist(x,mu)
					cluster = j
			if not cluster in cluster_dict.keys():
				cluster_dict[cluster]=[]
			cluster_dict[cluster].append(x)
		# maximization
		for j in cluster_dict.keys():
			mus[j]=arithmetic_mean(cluster_dict[j],gammas,len(cluster_dict[j]))

	# for i in cluster_dict.keys():
	# 	x=[]
	# 	y=[]
	# 	for j in cluster_dict[i]:
	# 		x.append(j[0])
	# 		y.append(j[1])
	# 	plt.scatter(x,y,s=5)
	
	# plt.savefig("k-means"+str(class_no)+".jpg")
	# plt.clf()
	return mus,cluster_dict

# to perform GMM algorithm
def GMM(xs,mus,cluster_dict,iter_count,class_no,ax=False,diag=False):
	gammas=[]
	for i in range(len(mus)):
		temp=[]
		for j in range(len(xs)):
			temp.append(1)
		gammas.append(temp)
	pis=[]
	for i in cluster_dict.keys():
		pis.append(len(cluster_dict[i])/len(xs))
	covs=[]
	for i in cluster_dict.keys():
		temp=[]
		for j in range(len(cluster_dict[i])):
			temp.append(1)
		covs.append(covariance(cluster_dict[i],mus[i],temp,len(cluster_dict[i])))
	
	for ii in range(iter_count):
		# expectation
		for i in range(len(xs)):
			denom=float(0)
			for j in range(len(mus)):
				# print(covs[j])
				# print(np.linalg.det(covs[j]))
				guass=guassian(mus[j],covs[j],xs[i])
				# distr=multivariate_normal(mean=mus[j],cov=covs[j])
				gammas[j][i]=pis[j]*guass
				denom+=pis[j]*guass
				# print(guass)
			# print(pis)
			# print(denom)
			for j in range(len(mus)):
				gammas[j][i]/=denom
		# maximization
		for i in range(len(mus)):
			Nk=float(0)
			for j in range(len(xs)):
				Nk+=gammas[i][j]
			pis[i]=Nk/len(xs)
			# print("###"+str(Nk))
			mus[i]=arithmetic_mean(xs,gammas[i],Nk)
			covs[i]=covariance(xs,mus[i],gammas[i],Nk)
		
	if(diag):
		for i in range(len(covs)):
			covs[i]=np.diag(np.diag(covs[i]))
	if(ax!=False):
		for i in range(len(mus)):
			distr=multivariate_normal(mean=mus[i],cov=covs[i])
			minx=cluster_dict[i][0][0]
			miny=cluster_dict[i][0][1]
			maxx=minx
			maxy=maxx
			for j in cluster_dict[i]:
				if(j[0]<minx):
					minx=j[0]
				if(j[0]>maxx):
					maxx=j[0]
				if(j[1]<miny):
					miny=j[1]
				if(j[1]>maxy):
					maxy=j[1]
			X=np.arange(minx-1,maxx+1,0.1)
			Y=np.arange(miny-1,maxy+1,0.1)
			X,Y=np.meshgrid(X,Y)
			pdf=np.zeros(X.shape)
			for j in range(X.shape[0]):
				for k in range(X.shape[1]):
					pdf[j,k]=distr.pdf([X[j,k],Y[j,k]])
			ax.contour(X,Y,pdf,5,cmap='binary')

	return mus,covs,pis

# helper function for the roc function for synthetic data
def data_for_roc(class_count,mus,covs,pis,dev_dict):
	data_set=[]
	y_values=[]
	for i in range(class_count):
		for j in range(len(dev_dict[i+1])):
			data_set.append(dev_dict[i+1][j])
			y_values.append(i+1)

	allclasses=[ [0 for j in range(class_count)] for i in range(len(data_set))]

	for i in range(len(data_set)):
		for j in range(class_count):
			k=len(mus[j])
			for l in range(k):
				# distr=multivariate_normal(mean=mus[j][l],cov=covs[j][l])
				allclasses[i][j]+=pis[j][l]*guassian(mus[j][l],covs[j][l],data_set[i])
	return y_values,allclasses

# helper function for the roc function for image data
def data_for_roc2(class_count,mus,covs,pis,data_set):
	allclasses=[ [0.0 for j in range(class_count)] for i in range(len(data_set))]
	for i in range(len(data_set)):
		for j in range(class_count):
			sum=0.0
			for k in range(len(data_set[i])):
				temp=0.0
				for l in range(len(mus[j])):
					# distr=multivariate_normal(mean=mus[j][l],cov=covs[j][l],seed=1000)
					# print(pis[j][l]*guassian(mus[j][l],covs[j][l],data_set[i][k]))
					temp+=pis[j][l]*guassian(mus[j][l],covs[j][l],data_set[i][k])
				# print(temp)
				temp+=1e-323
				sum+=math.log(temp)
			allclasses[i][j]=sum
	return allclasses

synthetic_train_file=open("train.txt",'r')
synthetic_dev_file=open("dev.txt",'r')

train_dict={}
for i in synthetic_train_file.readlines():
	j = list(map(float,i.split(',')))
	if not j[2] in train_dict.keys():
		train_dict[j[2]]=[]
	train_dict[j[2]].append([j[0],j[1]])

dev_dict={}
for i in synthetic_dev_file.readlines():
	j = list(map(float,i.split(',')))
	if not j[2] in dev_dict.keys():
		dev_dict[j[2]]=[]
	dev_dict[j[2]].append([j[0],j[1]])
xs=[]
ys=[]
for i in range(2):
	xs.extend(dev_dict[i+1])
	ys.extend([i+1 for j in range(len(dev_dict[i+1]))])
for k in [4,10]:
	mus=[]
	pis=[]
	covs=[]
	diag_covs=[]
	fig,ax_decision=plt.subplots(1,1,figsize=(20,20))
	for i in range(2):
		a,b=k_means(train_dict[i+1],k,10,i+1)
		a,b,c=GMM(train_dict[i+1],a,b,3,i+1,ax_decision,True)
		mus.append(a)
		covs.append(b)
		pis.append(c)
		temp=[]
		for j in b:
			temp.append(np.diag(np.diag(j)))
		diag_covs.append(temp)
	decision_boundary(mus,covs,pis,train_dict,ax_decision,"k = "+str(k)+" ")
	# a,b=data_for_roc(2,mus,covs,pis,dev_dict)
	# fig, [ax_roc,ax_det] = plt.subplots(1,2,figsize=(20,10))
	# title_str="k1= "+str(k)+" k2= "+str(k)+" "
	# fpr,fnr=roc_plots(a,b,ax_roc,"non-diagonal covariance"+title_str,2)
	# ax_roc.legend()
	# display = DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="non diagonal covariance"+title_str).plot(ax=ax_det)
	# a,b=data_for_roc(2,mus,diag_covs,pis,dev_dict)
	# title_str="k1= "+str(k)+" k2= "+str(k)+" "
	# fpr,fnr=roc_plots(a,b,ax_roc,"diagonal covariance"+title_str,2)
	# ax_roc.legend()
	# display = DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="diagonal covariance"+title_str).plot(ax=ax_det)
	# fig.savefig(title_str+"roc_and_det.jpg")
	# plt.close()
	accuracy(xs,ys,mus,covs,pis)





path = "/mnt/c/Users/srava/Desktop/sem6/PRML - CS5691/assignment3/"
folders = [path+"coast",path+"forest",path+"highway",path+"mountain",path+"opencountry"]
block_data_train=[]
block_data_dev=[]
for j in range(5):
	block_data_train.append([])
	block_data_dev.append([])
for id,i in enumerate(folders) :
	for j in os.listdir(i+"/train/"):
		temp_file=open(i+"/train/"+j,"r")
		for idx,k in enumerate(temp_file.readlines()):
			l = list(map(float,k[:-1].split(" ")))
			block_data_train[id].append(l)
	for j in os.listdir(i+"/dev/"):
		temp_file=open(i+"/dev/"+j,"r")
		temp=[]
		for idx,k in enumerate(temp_file.readlines()):
			l = list(map(float,k[:-1].split(" ")))
			temp.append(l)
		block_data_dev[id].append(temp)
fig, [ax_roc,ax_det] = plt.subplots(1,2,figsize=(20,10))
for k in [4]:
	mus=[]
	covs=[]
	pis=[]
	for i in range(5):
		a,b=k_means(block_data_train[i],k,10,i)
		# print(a,b)
		a,b,c=GMM(block_data_train[i],a,b,3,i)
		mus.append(a)
		covs.append(b)
		pis.append(c)
		# for j in range(len(b)):
		# 	print(np.linalg.det(b[j]))
	dev_data_set=[]
	yvalues=[]
	for i in range(5):
		dev_data_set.extend(block_data_dev[i])
		yvalues.extend([i for j in range(len(block_data_dev[i]))])
	temp=data_for_roc2(5,mus,covs,pis,dev_data_set)
	
	fpr,fnr=roc_plots(yvalues,temp,ax_roc,"K="+str(k),5)
	ax_roc.legend()
	display = DetCurveDisplay(fpr=fpr,fnr=fnr,estimator_name="K="+str(k)).plot(ax=ax_det)

fig.savefig("image_roc_and_det.jpg")
plt.close()
accuracy2(dev_data_set,yvalues,mus,covs,pis)