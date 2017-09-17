#	About: 
#		This program classifies the data for different classes by
#		assuming the case when Covariance Matrix = (sigma)^2*I
#		and is same for all classes.

import numpy as np
import math
import matplotlib.pyplot as plt

#	Variable:
#		classes - Array of training data for different classes
#		mean - Array of mean vectors of different classes
#		variance - Array of covariance matrices of different classes

classes=[]
classesRange=[]
testData=[]
mean=[]
variance=[]
dimension=2
confusionMatClass=[]
confusionMatrix=[]
constantDensityContours=[]
total_variance=0

#Funtion: calcPrereq

def calcPrereq(filename):
	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	tempClass=np.array(data)
	tempClassRange=[]
	tempClassRange.append(np.amin(tempClass,axis=0))
	tempClassRange.append(np.amax(tempClass,axis=0))
	data_train=[data[i] for i in range(long(0.75*len(tempClass)))]
	tempTestData=[data[i] for i in range(long(0.75*len(tempClass)),len(tempClass))]
	tempClassTrain=np.array(data_train)
	tempMean=tempClassTrain.mean(axis=0)
	tempVariance=[0,0]
	for j in range(len(tempMean)):
		sumj=0
		for i in range(len(tempClassTrain)):
			sumj+=(tempClassTrain[i][j]-tempMean[j])*(tempClassTrain[i][j]-tempMean[j]);
		tempVariance[j]=sumj/len(tempClassTrain)
	classes.append(tempClassTrain)
	mean.append(tempMean)
	variance.append(tempVariance)
	testData.append(tempTestData)
	classesRange.append(tempClassRange)

def gi(x):
	val=[0 for i in range(len(classes))]
	for i in range(len (classes)):
		val[i]=-1.0/2.0/total_variance;
		first_term=0;
		for j in range(dimension):
			first_term+=(x[j]-mean[i][j])*(x[j]-mean[i][j])
		val[i]*=first_term
		tot=0
		for j in range(len(classes)):
			tot+=len(classes[j])
		val[i]+=math.log(float(len(classes[i]))/tot)
	return np.argmax(val)

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
	val+=math.log(float(len(classes[first]))/len(classes[second]))
	if val<0:
		return first
	else:
		return second

def calcConfusion():
	global confusionMatrix
	confusionMatrix=[[0 for i in range(len(classes))] for i in range(len(classes))]
	for i in range(len(classes)):
		for j in range(len(testData[i])):
			x=testData[i][j]
			ret=gi(x)
			confusionMatrix[ret][i]+=1
	

def calcConfusionClass(ind):
	temp=[[0 for i in range(2)] for i in range(2)]
	for j in range(len(classes)):
		for i in range(len(testData[j])):
			x=testData[j][i]
			ret=gi(x)
			if ind==j:
				if ret==ind:
					temp[0][0]+=1
				else:
					temp[1][0]+=1
			else: 
				if ret==ind:
					temp[0][1]+=1
				else:
					temp[1][1]+=1
	confusionMatClass.append(temp)
	
print "\nThis program is a Baye's Classifier assuming the case when Covariance Matrix = (sigma)^2*I and is same for all classes.\n"

print "Enter which data you want to use : "
print "1. Linearly Separable Data."
print "2. Non-linearly Separable Data."
print "3. Real World Data."
choice=input("Choice : ")	
if(choice==1):
	calcPrereq("../../data/Input/ls_group2/Class1.txt")
	calcPrereq("../../data/Input/ls_group2/Class2.txt")
	calcPrereq("../../data/Input/ls_group2/Class3.txt")
elif(choice==2):
	calcPrereq("../../data/Input/nl_group2/Class1.txt")
	calcPrereq("../../data/Input/nl_group2/Class2.txt")
	calcPrereq("../../data/Input/nl_group2/Class3.txt")
elif(choice==3):
	calcPrereq("../../data/Input/rd_group2/Class1.txt")
	calcPrereq("../../data/Input/rd_group2/Class2.txt")
	calcPrereq("../../data/Input/rd_group2/Class3.txt")
else:
	print "Wrong input! Exiting.\n"

for i in range(len(classes)):
	for j in range(dimension):
		total_variance+=variance[i][j]
total_variance/=len(classes)*dimension

print "\nThe average variance calculated for all classes comes out to be",total_variance

print "\nThe mean and variance vectors for different classes are: \n"
for i in range(len(mean)):
	print "Class ",i+1,": Mean - ",mean[i]," Var - ",variance[i]

for i in range(len(classes)):
	calcConfusionClass(i)

calcConfusion()

Accuracy=[]
Precision=[]
Recall=[]
FMeasure=[]

print "\nThe Confusion Matrices for different classes are: "
for i in range(len(classes)):
	print "\nConfusion Matrix for class",i+1,": \n"
	print np.asmatrix(confusionMatClass[i])
	tp=confusionMatClass[i][0][0]
	fp=confusionMatClass[i][0][1]
	fn=confusionMatClass[i][1][0]
	tn=confusionMatClass[i][1][1]
	accuracy=float(tp+tn)/(tp+tn+fp+fn)
	precision=float(tp)/(tp+fp)
	recall=float(tp)/(tp+fn)
	fMeasure=2*precision*recall/(precision+recall)
	print "\nClassification Accuracy for class",i+1,"is",accuracy
	print "Precision for class",i+1,"is",precision
	print "Recall for class",i+1,"is",recall
	print "F-measure for class",i+1,"is",fMeasure
	Accuracy.append(accuracy),Precision.append(precision),Recall.append(recall),FMeasure.append(fMeasure)

avgAccuracy,avgPrecision,avgRecall,avgFMeasure=0,0,0,0
for i in range (len(classes)):
	avgAccuracy+=Accuracy[i]
	avgPrecision+=Precision[i]
	avgRecall+=Recall[i]
	avgFMeasure+=FMeasure[i]
avgAccuracy/=len(classes)
avgPrecision/=len(classes)
avgRecall/=len(classes)
avgFMeasure/=len(classes)

print "\nThe Confusion Matrix of all classes together is: \n"
print np.asmatrix(confusionMatrix)
print "\nAverage classification Accuracy is",avgAccuracy
print "Average precision is",avgPrecision
print "Average recall is",avgRecall
print "Average F-measure is",avgFMeasure

print "\nPlease wait for a minute or two while the program generates graphs..."

colors=['b','g','r']
colorsTestData=['c','m','y']

for i in range(len(classes)):
	circles=[]
	for j in range(4):
		circles.append(plt.Circle((mean[i][0],mean[i][1]),total_variance/float(j+1),color='black',fill=False))
	constantDensityContours.append(circles)

l=1
f=[]

f.append(plt.figure(l))
l+=1
minArr=[0 for i in range(dimension)]
maxArr=[0 for i in range(dimension)]
for i in range(dimension):
	minArr[i]=classesRange[0][0][i]
	maxArr[i]=classesRange[0][1][i]

for i in range(len(classesRange)):
	for j in range(dimension):
		if(minArr[j]>classesRange[i][0][j]):
			minArr[j]=classesRange[i][0][j]
		if(maxArr[j]<classesRange[i][1][j]):
			maxArr[j]=classesRange[i][1][j]

plt.subplot(111)
xRange=np.arange(minArr[0],maxArr[0],float(maxArr[0]-minArr[0])/100)
yRange=np.arange(minArr[1],maxArr[1],float(maxArr[1]-minArr[1])/100)
for i in range(len(xRange)):
	for j in range(len(yRange)):
		X=[0,0]
		X[0]=xRange[i];
		X[1]=yRange[j];
		plt.plot(xRange[i],yRange[j],'.',color=colors[gi(X)])
for j in range(len(classes)):
	plt.plot([classes[j][i][0] for i in range(len(classes[j]))],[classes[j][i][1] for i in range(len(classes[j]))],'o',color=colorsTestData[j],label='Class {i}'.format(i=j))
f[l-2].suptitle("Decision Region plot for all Classes")


for j in range(len(classes)):
	for k in range(j+1,len(classes)):
		f.append(plt.figure(l))
		l+=1
		minArr=[0 for i in range(dimension)]
		maxArr=[0 for i in range(dimension)]
		for i in range(dimension):
			minArr[i]=classesRange[j][0][i]
			maxArr[i]=classesRange[j][1][i]
		for i in range(dimension):
			if(minArr[i]>classesRange[k][0][i]):
				minArr[i]=classesRange[k][0][i]
			if(maxArr[i]<classesRange[k][1][i]):
				maxArr[i]=classesRange[k][1][i]
		plt.subplot(111)
		xRange=np.arange(minArr[0],maxArr[0],float(maxArr[0]-minArr[0])/100)
		yRange=np.arange(minArr[1],maxArr[1],float(maxArr[1]-minArr[1])/100)
		for m in range(len(xRange)):
			for n in range(len(yRange)):
				X=[0,0]
				X[0]=xRange[m];
				X[1]=yRange[n];
				plt.plot(xRange[m],yRange[n],'.',color=colors[g(X,j,k)])
			plt.plot([classes[j][i][0] for i in range(len(classes[j]))],[classes[j][i][1] for i in range(len(classes[j]))],'o',color=colorsTestData[j],label='Class {i}'.format(i=j))
			plt.plot([classes[k][i][0] for i in range(len(classes[k]))],[classes[k][i][1] for i in range(len(classes[k]))],'o',color=colorsTestData[k],label='Class {i}'.format(i=k))
		label="Decision Region plot for class pair ("+str(j+1)+","+str(k+1)+")"
		f[l-2].suptitle(label)

for i in range(len(f)):
	f[i].show()

g=plt.figure(5)
for j in range(len(classes)):
	plt.plot([classes[j][i][0] for i in range(len(classes[j]))],[classes[j][i][1] for i in range(len(classes[j]))],'.',color=colors[j],label='Class {i}'.format(i=j))
for i in range(len(constantDensityContours)):	
	for j in range(4):
		plt.gca().add_patch(constantDensityContours[i][j])

g.suptitle("Constant Density Contours for all classes")
plt.axis('scaled')
g.show()

plt.show()

