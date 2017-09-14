import numpy as np
import math
import matplotlib.pyplot as plt


####################################################################################### 1st Class ############################################################################################
file1=open("../../data/LS_Group2/Class1.txt")
data = []
for line in file1:
    number_strings = line.split() #splits across whitespaces
    numbers = [float(n) for n in number_strings]
    data.append(numbers)

class1=np.array(data)

data_train=[data[i] for i in range(len(class1))]

class1_train=np.array(data_train)
mean1=class1_train.mean(axis=0)

variance1x=math.sqrt(sum((class1_train[i][0]-mean1[0])*(class1_train[i][0]-mean1[0]) for i in range(len(class1_train)))/len(class1))
variance1y=math.sqrt(sum((class1_train[i][1]-mean1[1])*(class1_train[i][1]-mean1[1]) for i in range(len(class1_train)))/len(class1))



####################################################################################### 2nd Class ############################################################################################
file2=open("../../data/LS_Group2/Class2.txt")
data = []
for line in file2:
    number_strings = line.split()#splits across whitespaces
    numbers = [float(n) for n in number_strings]
    data.append(numbers)

class2=np.array(data)

data_train=[data[i] for i in range(len(class2))]

class2_train=np.array(data_train)
mean2=class2_train.mean(axis=0)

variance2x=math.sqrt(sum((class2_train[i][0]-mean2[0])*(class2_train[i][0]-mean2[0]) for i in range(len(class2_train)))/len(class2))
variance2y=math.sqrt(sum((class2_train[i][1]-mean2[1])*(class2_train[i][1]-mean2[1]) for i in range(len(class2_train)))/len(class2))


####################################################################################### 3rd Class ############################################################################################
file3=open("../../data/LS_Group2/Class3.txt")
data = []
for line in file3:
    number_strings = line.split() #splits across whitespaces
    numbers = [float(n) for n in number_strings]
    data.append(numbers)
class3=np.array(data)

data_train=[data[i] for i in range(len(class3))]

class3_train=np.array(data_train)
mean3=class3_train.mean(axis=0)

variance3x=math.sqrt(sum((class3_train[i][0]-mean3[0])*(class3_train[i][0]-mean3[0]) for i in range(len(class3_train)))/len(class3))
variance3y=math.sqrt(sum((class3_train[i][1]-mean3[1])*(class3_train[i][1]-mean3[1]) for i in range(len(class3_train)))/len(class3))


###############################################################################################################################################################################################


sigma=np.matrix([[(variance1x*variance1x + variance2x*variance2x)/2,0],[0,(variance2x*variance2x + variance2y*variance2y)/2]])
sigma_inverse=sigma.I
print(sigma_inverse)


omega_=np.matmul(sigma_inverse,mean1-mean2)
print(omega_)

plt.subplot(331)
plt.plot([class1[i][0] for i in range(len(class1))],[class1[i][1] for i in range(len(class1))],'o')
plt.plot([class2[i][0] for i in range(len(class2))],[class2[i][1] for i in range(len(class2))],'o')
plt.plot([class3[i][0] for i in range(len(class3))],[class3[i][1] for i in range(len(class3))],'o')


plt.subplot(334)
plt.plot([class1[i][0] for i in range(len(class1))],[class1[i][1] for i in range(len(class1))],'o')
plt.plot([class2[i][0] for i in range(len(class2))],[class2[i][1] for i in range(len(class2))],'o')
plt.plot([class3[i][0] for i in range(len(class3))],[class3[i][1] for i in range(len(class3))],'o')


plt.show()

#we have to take 75% data only
