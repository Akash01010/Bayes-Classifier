#	About: 
#		This program classifies the data for different classes by
#		assuming the case when Covariance Matrix = (sigma)^2*I

import numpy as np
import math
import matplotlib.pyplot as plt

#	Variable:
#		classes - Array of training data for different classes
#		mean - Array of mean vectors of different classes
#		variance - Array of covariance matrix of different classes

classes=[]
testData=[]
mean=[]
variance=[]
lenClasses=[]
dimension=2
confusionMat=[]
constantDensityContours=[]
total_variance=0

#Funtion: calcPrereq
#	

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
	tempTestData=[data[i] for i in range(long(0.75*len(tempClass))+1,len(tempClass))]
	tempClassTrain=np.array(data_train)
	tempMean=tempClassTrain.mean(axis=0)
	tempVariance=[]
	for j in range(len(tempMean)):
		tempVariance.append(sum((tempClassTrain[i][j]-tempMean[j])*(tempClassTrain[i][j]-tempMean[j]) for i in range(len(tempClassTrain)))/len(tempClassTrain))
	classes.append(tempClassTrain)
	mean.append(tempMean)
	variance.append(tempVariance)
	lenClasses.append(len(tempClassTrain))
	testData.append(tempTestData)

def g(x,first,second):
	val=1.0/2.0/total_variance;
	first_term=0;
	for i in range(dimension):
		first_term+=x[i]*(mean[second][i]-mean[first][i])
	first_term*=2
	second_term=0;
	for i in range(dimension):
		second_term+=(mean[first][i]*mean[first][i])-(mean[second][i]*mean[second][i])
	val*=first_term+second_term
	if val<0:
		return first
	else:
		return second

def calcConfusion(first,second):
	temp=[[0 for i in range(2)] for i in range(2)]
	for i in range(len(testData[first])):
		x=testData[first][i]
		ret=g(x,first,second)
		if ret==first:
			temp[0][0]+=1
		else:
			temp[0][1]+=1
	for i in range(len(testData[second])):
		x=testData[second][i]
		ret=g(x,second,first)
		if ret==second:
			temp[1][1]+=1
		else:
			temp[1][0]+=1
	confusionMat.append(temp)
	
	
calcPrereq("../../data/Input/rd_group2/Class1.txt")
calcPrereq("../../data/Input/rd_group2/Class2.txt")
calcPrereq("../../data/Input/rd_group2/Class3.txt")


for i in range(len(classes)):
	for j in range(dimension):
		total_variance+=variance[i][j]
total_variance/=len(classes)*dimension
sigma=np.matrix(total_variance*np.identity(dimension))


for i in range(len(classes)):
	for j in range(i+1,len(classes)):
		calcConfusion(i,j)

print(mean)

print(confusionMat)

colors=['b','g','r']

for i in range(len(classes)):
	circles=[]
	for j in range(4):
		circles.append(plt.Circle((mean[i][0],mean[i][1]),total_variance/float(j+1),color=colors[i],fill=False))
	constantDensityContours.append(circles)


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
