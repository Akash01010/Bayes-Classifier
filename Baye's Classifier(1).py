import numpy as np
import math
import matplotlib.pyplot as plt


file1=open("LS_Group2/Class1.txt")
data = []
for line in file1:
    number_strings = line.split() #splits across whitespaces
    numbers = [float(n) for n in number_strings]
    data.append(numbers)

#class1.mean(axis=0)
class1=np.array(data)
#print (class1)

mean1=class1.mean(axis=0)

variance1x=math.sqrt(sum((class1[i][0]-mean1[0])*(class1[i][0]-mean1[0]) for i in range(len(class1))))
variance1y=math.sqrt(sum((class1[i][1]-mean1[1])*(class1[i][1]-mean1[1]) for i in range(len(class1))))


file2=open("LS_Group2/Class2.txt")
data = []
for line in file2:
    number_strings = line.split() #splits across whitespaces
    numbers = [float(n) for n in number_strings]
    data.append(numbers)

#class1.mean(axis=0)
class2=np.array(data)
#print (class2)

mean2=class2.mean(axis=0)

variance2x=math.sqrt(sum((class2[i][0]-mean2[0])*(class2[i][0]-mean2[0]) for i in range(len(class2))))
variance2y=math.sqrt(sum((class2[i][1]-mean1[1])*(class2[i][1]-mean2[1]) for i in range(len(class2))))


file3=open("LS_Group2/Class3.txt")
data = []
for line in file3:
    number_strings = line.split() #splits across whitespaces
    numbers = [float(n) for n in number_strings]
    data.append(numbers)

#class1.mean(axis=0)
class3=np.array(data)
#print (class3)

mean3=class3.mean(axis=0)

variance3x=math.sqrt(sum((class3[i][0]-mean3[0])*(class3[i][0]-mean3[0]) for i in range(len(class3))))
variance3y=math.sqrt(sum((class3[i][1]-mean3[1])*(class3[i][1]-mean3[1]) for i in range(len(class3))))


var1=(variance1x+variance1y)/2
var2=(variance2x+variance2y)/2
var3=(variance3x+variance3y)/2

#varx=(variance1x+variance2x+variance3x)/3
#vary=(variance1y+variance2y+variance3y)/3

sigma=np.matrix([[(var1*var1 + var2*var2 +var3*var3)/3,0],[0,(var1*var1 + var2*var2 +var3*var3)/3]])
sigma_inverse=sigma.I
print(sigma_inverse)

#print(sum(sum(class1[i] for i in range(1,len(class1))))/float(len(class1)))
plt.plot([class1[i][0] for i in range(len(class1))],[class1[i][1] for i in range(len(class1))],'o')
plt.plot([class2[i][0] for i in range(len(class2))],[class2[i][1] for i in range(len(class2))],'o')
plt.plot([class3[i][0] for i in range(len(class3))],[class3[i][1] for i in range(len(class3))],'o')




#plt.show()