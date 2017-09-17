#	About: 
#		This program classifies the data for different classes by
#		assuming the case when all classes have full but equal
#		Covariance Matrices.

import numpy as np
import math
import matplotlib.pyplot as plt

#	Variable:
#		classes - Array of training data for different classes
#		mean - Array of mean vectors of different classes
#		variance - Array of covariance matrices of different classes

classes=[]
testData=[]
mean=[]
covarianceMatrices=[]
dimension=2
covarianceMatrix=np.zeros(shape=(dimension,dimension))
covarianceMatrixInv=np.zeros(shape=(dimension,dimension))
confusionMat=[]
constantDensityContours=[]

#Funtion: calcPrereq

def calcPrereq(filename):
	global classes,mean,variance

	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)

	tempClass=np.array(data)
	data_train=[data[i] for i in range(long(0.75*len(tempClass)))]
	tempTestData=[data[i] for i in range(long(0.75*len(tempClass)),len(tempClass))]
	tempClassTrain=np.array(data_train)
	tempMean=tempClassTrain.mean(axis=0)
	# tempVariance=[]
	# for j in range(len(tempMean)):
	# 	tempVariance.append(sum((tempClassTrain[i][j]-tempMean[j])*(tempClassTrain[i][j]-tempMean[j]) for i in range(len(tempClassTrain)))/len(tempClassTrain))
	classes.append(tempClassTrain)
	mean.append(tempMean)
	# variance.append(tempVariance)
	testData.append(tempTestData)

# def g(x,first,second):
# 	val=1.0/2.0/total_variance;
# 	first_term=0;
# 	for i in range(dimension):
# 		first_term+=x[i]*(mean[second][i]-mean[first][i])
# 	first_term*=2
# 	second_term=0;
# 	for i in range(dimension):
# 		second_term+=(mean[first][i]*mean[first][i])-(mean[second][i]*mean[second][i])
# 	val*=first_term+second_term
# 	if val<0:
# 		return first
# 	else:
# 		return second

# def calcConfusion(first,second):
# 	temp=[[0 for i in range(2)] for i in range(2)]
# 	for i in range(len(testData[first])):
# 		x=testData[first][i]
# 		ret=g(x,first,second)
# 		if ret==first:
# 			temp[0][0]+=1
# 		else:
# 			temp[0][1]+=1
# 	for i in range(len(testData[second])):
# 		x=testData[second][i]
# 		ret=g(x,second,first)
# 		if ret==second:
# 			temp[1][1]+=1
# 		else:
# 			temp[1][0]+=1
# 	confusionMat.append(temp) 

def exp(ind,i,j):
	sum=0
	for k in range(len(classes[ind])):
		x=classes[ind][k]
		sum+=(x[i]-mean[ind][i])*(x[j]-mean[ind][j])
	sum/=len(classes[ind])
	return sum

def calcCovarianceMat():
	for i in range(len(classes)):
		tempCovarianceMat=np.zeros(shape=(dimension,dimension))
		for j in range(dimension):
			for k in range(dimension):
				tempCovarianceMat[j,k]=exp(i,j,k)
		covarianceMatrices.append(tempCovarianceMat)
	for i in range(dimension):
		for j in range(dimension):
			sum=0
			for k in range(len(covarianceMatrices)):
				sum+=covarianceMatrices[k][i,j]
			sum/=len(covarianceMatrices)
			covarianceMatrix[i,j]=sum


calcPrereq("../../data/Input/nl_group2/Class1.txt")
calcPrereq("../../data/Input/nl_group2/Class2.txt")
calcPrereq("../../data/Input/nl_group2/Class3.txt")
calcCovarianceMat()		
covarianceMatrixInv=np.asmatrix(covarianceMatrix).I


# for i in range(len(classes)):
# 	for j in range(dimension):
# 		total_variance+=variance[i][j]
# total_variance/=len(classes)*dimension
# sigma=np.matrix(total_variance*np.identity(dimension))


# for i in range(len(classes)):
# 	for j in range(i+1,len(classes)):
# 		calcConfusion(i,j)

print mean

print confusionMat

print covarianceMatrix

print covarianceMatrixInv

print np.matmul(covarianceMatrix,covarianceMatrixInv)

colors=['b','g','r']

# for i in range(len(classes)):
# 	circles=[]
# 	for j in range(4):
# 		circles.append(plt.Circle((mean[i][0],mean[i][1]),covarianceMatrix[i][i]/float(j+1),color=colors[i],fill=False))
# 	constantDensityContours.append(circles)


# plt.subplot(331)
# plt.plot([classes[0][i][0] for i in range(len(classes[0]))],[classes[0][i][1] for i in range(len(classes[0]))],'.')
# # for i in range(4):
# # 	plt.gca().add_patch(constantDensityContours[0][i])

# plt.subplot(332)
# plt.plot([classes[1][i][0] for i in range(len(classes[1]))],[classes[1][i][1] for i in range(len(classes[1]))],'.')
# # for i in range(4):
# # 	plt.gca().add_patch(constantDensityContours[1][i])


# plt.subplot(333)
# plt.plot([classes[2][i][0] for i in range(len(classes[2]))],[classes[2][i][1] for i in range(len(classes[2]))],'.')
# # for i in range(4):
# # 	plt.gca().add_patch(constantDensityContours[2][i])


# plt.subplot(334)
# plt.plot([classes[0][i][0] for i in range(len(classes[0]))],[classes[0][i][1] for i in range(len(classes[0]))],'.')
# plt.plot([classes[1][i][0] for i in range(len(classes[1]))],[classes[1][i][1] for i in range(len(classes[1]))],'.')

# plt.subplot(335)
# plt.plot([classes[0][i][0] for i in range(len(classes[0]))],[classes[0][i][1] for i in range(len(classes[0]))],'.')
# plt.plot([classes[2][i][0] for i in range(len(classes[2]))],[classes[2][i][1] for i in range(len(classes[2]))],'.')

# plt.subplot(336)
# plt.plot([classes[1][i][0] for i in range(len(classes[1]))],[classes[1][i][1] for i in range(len(classes[1]))],'.')
# plt.plot([classes[2][i][0] for i in range(len(classes[2]))],[classes[2][i][1] for i in range(len(classes[2]))],'.')

plt.subplot(111)
plt.plot([classes[0][i][0] for i in range(len(classes[0]))],[classes[0][i][1] for i in range(len(classes[0]))],'.')
plt.plot([classes[1][i][0] for i in range(len(classes[1]))],[classes[1][i][1] for i in range(len(classes[1]))],'.')
plt.plot([classes[2][i][0] for i in range(len(classes[2]))],[classes[2][i][1] for i in range(len(classes[2]))],'.')

plt.axis('scaled')
plt.show()

#we have to take 75% data only
