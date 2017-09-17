import numpy as np
import math
import matplotlib.pyplot as plt

classes=[]
classesRange=[]
testData=[]
mean=[]
covarianceMatrices=[]
dimension=2
covarianceMatricesInv=[]
confusionMatrix=[]
confusionMatClass=[]

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
	classes.append(tempClassTrain)
	mean.append(tempMean)
	testData.append(tempTestData)
	classesRange.append(tempClassRange)

def gi(x):
	val=[0 for i in range(len(classes))]
	for i in range(len (classes)):
		first_term=0
		tempFirstTerm=np.zeros(shape=(1,dimension))
		for j in range(dimension):
			for k in range(dimension):
				tempFirstTerm[0,j]+=x[k]*covarianceMatricesInv[i][k,j]
		for j in range(dimension):
			first_term+=tempFirstTerm[0,j]*x[j]
		first_term*=-0.5
		second_term=0
		tempSecondTerm=np.zeros(shape=(dimension,1))
		for j in range(dimension):
			for k in range(dimension):
				tempSecondTerm[j,0]+=covarianceMatricesInv[i][j,k]*mean[i][k]
		third_term=0
		for j in range(dimension):
			second_term+=tempSecondTerm[j,0]*x[j]
			third_term+=tempSecondTerm[j,0]*mean[i][j]
		third_term+math.log(np.linalg.det(covarianceMatrices[i]))
		third_term*=-0.5
		tot=0
		for j in range(len(classes)):
			tot+=len(classes[j])
		third_term+=math.log(float(len(classes[i]))/tot)
		val[i]=first_term+second_term+third_term
	return np.argmax(val)

def classVal(x,ind):
	first_term=0
	tempFirstTerm=np.zeros(shape=(1,dimension))
	for j in range(dimension):
		for k in range(dimension):
			tempFirstTerm[0,j]+=x[k]*covarianceMatricesInv[ind][k,j]
	for j in range(dimension):
		first_term+=tempFirstTerm[0,j]*x[j]
	first_term*=-0.5
	second_term=0
	tempSecondTerm=np.zeros(shape=(dimension,1))
	for j in range(dimension):
		for k in range(dimension):
			tempSecondTerm[j,0]+=covarianceMatricesInv[ind][j,k]*mean[ind][k]
	third_term=0
	for j in range(dimension):
		second_term+=tempSecondTerm[j,0]*x[j]
		third_term+=tempSecondTerm[j,0]*mean[ind][j]
	third_term+math.log(np.linalg.det(covarianceMatrices[ind]))
	third_term*=-0.5
	tot=0
	for j in range(len(classes)):
		tot+=len(classes[j])
	third_term+=math.log(float(len(classes[ind]))/tot)
	return first_term+second_term+third_term	

def g(x,first,second):
	if classVal(x,first)-classVal(x,second)<0:
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
				if(j==k):
					tempCovarianceMat[j,k]=exp(i,j,k)
				else:
					tempCovarianceMat[j,k]=0
		covarianceMatrices.append(tempCovarianceMat)
	for i in range(len(classes)):
		covarianceMatricesInv.append(np.asmatrix(covarianceMatrices[i]).I)


print( "\nThis program is a Baye's Classifier assuming the case when all classes have full but equal Covariance Matrices.\n")

print( "Enter which data you want to use : ")
print( "1. Linearly Separable Data.")
print( "2. Non-linearly Separable Data.")
print( "3. Real World Data.")
choice=input("Choice : ")	

if(choice==1):
	calcPrereq("../../data/ls_group2/Class1.txt")
	calcPrereq("../../data/ls_group2/Class2.txt")
	calcPrereq("../../data/ls_group2/Class3.txt")
elif(choice==2):
	calcPrereq("../../data/nl_group2/Class1.txt")
	calcPrereq("../../data/nl_group2/Class2.txt")
	calcPrereq("../../data/nl_group2/Class3.txt")
elif(choice==3):
	calcPrereq("../../data/rd_group2/Class1.txt")
	calcPrereq("../../data/rd_group2/Class2.txt")
	calcPrereq("../../data/rd_group2/Class3.txt")
else:
	print( "Wrong input! Exiting.\n")

choices=['ls','nl','rd']

calcCovarianceMat()

print( "\nThe mean vector and Covariance matrices for different classes are: \n")

for i in range(len(classes)):
	print( "Class ",i+1,":")
	print( "\nMean - ",mean[i])
	print( "\nCovariance Matrix - ")
	print( covarianceMatrices[i],"\n")

for i in range(len(classes)):
	calcConfusionClass(i)

calcConfusion()

Accuracy=[]
Precision=[]
Recall=[]
FMeasure=[]

print( "\nThe Confusion Matrices for different classes are: ")
for i in range(len(classes)):
	print( "\nConfusion Matrix for class",i+1,": \n")
	print( np.asmatrix(confusionMatClass[i]))
	tp=confusionMatClass[i][0][0]
	fp=confusionMatClass[i][0][1]
	fn=confusionMatClass[i][1][0]
	tn=confusionMatClass[i][1][1]
	accuracy=float(tp+tn)/(tp+tn+fp+fn)
	precision=float(tp)/(tp+fp)
	recall=float(tp)/(tp+fn)
	fMeasure=2*precision*recall/(precision+recall)
	print( "\nClassification Accuracy for class",i+1,"is",accuracy)
	print( "Precision for class",i+1,"is",precision)
	print( "Recall for class",i+1,"is",recall)
	print( "F-measure for class",i+1,"is",fMeasure)
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

print( "\nThe Confusion Matrix of all classes together is: \n")
print( np.asmatrix(confusionMatrix))
print( "\nAverage classification Accuracy is",avgAccuracy)
print( "Average precision is",avgPrecision)
print( "Average recall is",avgRecall)
print( "Average F-measure is",avgFMeasure)

print( "\nPlease wait for a minute or two while the program generates graphs...")

colors=['b','g','r']
colorsTestData=['c','m','y']

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
f[l-2].savefig('../../data/Output/C_AllClasses_DR_'+choices[choice-1]+'.png')


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
		f[l-2].savefig('../../data/Output/C_ClassPair_'+str(j+1)+'_'+str(k+1)+'_DR_'+choices[choice-1]+'.png')

for i in range(len(f)):
	f[i].show()

g=plt.figure(5)
for j in range(len(classes)):
	ax=plt.subplot(111)
	plt.plot([classes[j][i][0] for i in range(len(classes[j]))],[classes[j][i][1] for i in range(len(classes[j]))],'.',color=colors[j],label='Class {i}'.format(i=j))
	u=[]
	for k in range(dimension):
		tempU=np.linspace(classesRange[j][0][k],classesRange[j][1][k],10)
		u.append(tempU)
	x,y=np.meshgrid(u[0],u[1]) 
	temp=-0.5*covarianceMatricesInv[j]
	temp1=np.matmul(covarianceMatricesInv[j],mean[j])
	const=np.matmul(np.matmul(mean[j].transpose(),temp),mean[j])
	tot=0
	for j in range(len(classes)):
		tot+=len(classes[j])
	constant=const[0,0]-0.5*math.log(np.linalg.det(covarianceMatrices[j]))+math.log(float(len(classes[j]))/tot)
	z=(temp[0,0])*(x**2)+2*(temp[0,1])*x*y+temp[1,1]*(y**2)+temp1[0,0]*x+temp1[0,1]*y+constant
	ax.contour(x,y,z)

g.suptitle("Constant Density Contours for all classes")
g.savefig('../../data/Output/C_AllClasses_CDC_'+choices[choice-1]+'.png')
plt.axis('scaled')
g.show()

plt.show()