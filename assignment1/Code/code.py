import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from PIL import Image, ImageOps
from matplotlib import cm
import shutil
import os
#Read the image on whose decompositions are to be performed
img = np.array(Image.open('40.jpg'))
n,m = img.shape
#Function Purpose: To save the matrix as grapy scale image at given address
#Args : 
# 	1) address: This is a string that contains address where the image has to be saved
# 	2) cimg:	A matrix which is to be saved as an image
#Return Value: N/A
def writeToFile(address,cimg):
	if os.path.exists(address):
		os.remove(address)
	mplimg.imsave(address,np.uint8(cimg),cmap=cm.gray)
#Function Purpose: To get the Singular value Decomposition of a matrix using EigenValue decomposition
#Args :
# 	1) M : The matrix on which SVD has to be performed.
#Return Value:	Tuple whose First value is U in decomposition,Second value contains numpy array of singular values,
#				Third value contains V* in decomposition.
def getSingular(M):
	tmpM = np.array(M,dtype=complex)
	tmpMt = np.transpose(tmpM)
	e1val,U = np.linalg.eig(tmpM @ tmpMt)						#Get the value of U from MM^T
	ind1 = list(range(n))
	ind1.sort(key=lambda i:e1val[i],reverse=True)				#Sort and rearrange columns of U according to magnitude of eigenvalues found above
	egval = np.zeros(n,dtype=complex)
	newU = np.zeros((n,n),dtype=complex)
	for i in range(n):
		egval[i] = e1val[ind1[i]]
	for i in range(n):
		for j in range(n):
			newU[i][j] = U[i][ind1[j]]
	sig = np.sqrt(egval)
	newVt =  np.diag((1.0/sig)) @ np.linalg.inv(newU) @ tmpM    #Find V* from M, U and singular values.
	return newU,sig,newVt										#End of function, return the calculated values.

evals,evec = np.linalg.eig(img)									#Using numpy linalg function eig, find the EVD of image.							
eigPairs = []
vis = np.zeros(n,dtype=bool)
for i in range(n):												#These nested loops find and pair eigenvalues with their conjugates.
	if vis[i]:
		continue
	if abs(evals[i].imag) < 1e-8:
		eigPairs.append((i,-1))
		continue
	f = False
	for j in range(i+1,n):
		if not vis[j] and abs(np.conj(evals[i])-evals[j]) < 1e-8:
			vis[j] = True
			eigPairs.append((i,j))
			f = True
			break
	assert f
eigPairs.sort(key=lambda v: abs(evals[v[0]]),reverse=True)		#Sort eigenvalue pairs according to their magnitude
ievec = np.linalg.inv(evec)
lstkEvd = []
lstkSvd = []
lstfNormEVD = []												#These lists store value of k and frobenius norms to plot graph later
lstfNormSVD = []
U,sval,Vt = getSingular(img)									#Perform svd by calling above function
sig = np.diag(sval)
curK = 0
D = np.zeros((n,n),dtype=complex)
P = np.zeros((n,n),dtype=complex)								#These matrices are P , D , P^-1 which will have rearranged values according to magnitudes.
iP = np.zeros((n,n),dtype=complex)
for p in eigPairs:
	D[curK][curK] = evals[p[0]]
	for j in range(n):
		P[j][curK] = evec[j][p[0]]
		iP[curK][j] = ievec[p[0]][j]							#Rearranging values in these nested loops
	curK += 1
	if p[1] != -1:
		D[curK][curK] = evals[p[1]]
		for j in range(n):
			P[j][curK] = evec[j][p[1]]
			iP[curK][j] = ievec[p[1]][j]
		curK += 1
if os.path.exists("SVDImages"):
	shutil.rmtree("SVDImages")								
os.makedirs("SVDImages")
if not os.path.exists("SVDImages/Reconstructed"):				#Create some folders to save reconstructed images and error images in different decompositions
	os.makedirs("SVDImages/Reconstructed")
if not os.path.exists("SVDImages/Error"):
	os.makedirs("SVDImages/Error")
if os.path.exists("EVDImages"):
	shutil.rmtree("EVDImages")
os.makedirs("EVDImages")
if not os.path.exists("EVDImages/Reconstructed"):
	os.makedirs("EVDImages/Reconstructed")
if not os.path.exists("EVDImages/Error"):
	os.makedirs("EVDImages/Error")
curK = 0
for p in eigPairs:												#Loop to save various reconstructed images of EVD for varying k
	if p[1] == -1:
		curK += 1
	else:
		curK += 2
	print("Progress", curK,"/",2*n)
	lstkEvd.append(curK)
	curD = D[:curK,:curK]
	curP = P[:,:curK]
	curiP = iP[:curK,:]
	icurImg = curP @ curD @ curiP
	assert np.all(icurImg.imag < 1e-8)
	curImg = abs(icurImg)
	errImg = img - curImg
	file = 'EVDImages/Reconstructed/k' + (str(curK)+'.jpg')
	writeToFile(file,curImg)
	file2 = 'EVDImages/Error/k' + (str(curK)+'.jpg')
	writeToFile(file2,errImg)
	frobNorm = math.sqrt(np.sum(np.square(errImg)))
	lstfNormEVD.append(frobNorm)
for k in range(1,257):
	print("Progress", k + n,"/",2*n)
	lstkSvd.append(k)
	curU = U[:,:k]
	curS = sig[:k,:k]
	curVt = Vt[:k,:]
	icurImg = curU @ curS @ curVt
	curImg = abs(icurImg)
	errImg = img - curImg
	file = 'SVDImages/Reconstructed/k' + (str(k)+'.jpg')
	writeToFile(file,curImg)
	file2 = 'SVDImages/Error/k' + (str(k)+'.jpg')
	writeToFile(file2,errImg) 
	frobNorm = math.sqrt(np.sum(np.square(errImg)))
	lstfNormSVD.append(frobNorm)
plt.plot(lstkEvd,lstfNormEVD,label='EVD')
plt.plot(lstkSvd,lstfNormSVD,label='SVD')
plt.xlabel("Value of K")
plt.ylabel("Frobenius norm of error Image")
plt.legend(loc='best')
graphF = "graph.jpg"
if os.path.exists(graphF):
	os.remove(graphF)
plt.savefig(graphF,bbox_inches='tight')


#Commented snippet that plots images into a single file side by side to compare for some values of k given to it(Here:  6, 15, 30,60,151)


# tmp,ax = plt.subplots(5,4,figsize=(10,20))
# for idx,k in enumerate([6,15,30,60,151]):
# 	evdImg = np.array(P[:,:k] @ D[:k,:k] @ iP[:k,:],dtype=float)
# 	svdImg = np.array(U[:,:k] @ sig[:k,:k] @ Vt[:k,:],dtype=float)
# 	ax[idx][0].imshow(evdImg,cmap='gray')
# 	ax[idx][0].set_title("evd for k = " + str(k))
# 	ax[idx][1].imshow(img-evdImg,cmap='gray')
# 	ax[idx][1].set_title("evd error for k = " + str(k))
# 	ax[idx][2].imshow(svdImg,cmap='gray')
# 	ax[idx][2].set_title("svd for k = " + str(k))
# 	ax[idx][3].imshow(img-svdImg,cmap='gray')
# 	ax[idx][3].set_title("svd error for k = " + str(k))
# plt.savefig("comparImg.jpg")