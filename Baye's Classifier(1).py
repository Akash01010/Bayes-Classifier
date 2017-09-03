import numpy as np
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

print (class1.mean(axis=0))

file2=open("LS_Group2/Class2.txt")
data = []
for line in file2:
    number_strings = line.split() #splits across whitespaces
    numbers = [float(n) for n in number_strings]
    data.append(numbers)

#class1.mean(axis=0)
class2=np.array(data)
#print (class2)

print (class2.mean(axis=0))

file3=open("LS_Group2/Class3.txt")
data = []
for line in file3:
    number_strings = line.split() #splits across whitespaces
    numbers = [float(n) for n in number_strings]
    data.append(numbers)

#class1.mean(axis=0)
class3=np.array(data)
#print (class3)

print (class3.mean(axis=0))

#print(sum(sum(class1[i] for i in range(1,len(class1))))/float(len(class1)))
plt.plot([class1[i][0] for i in range(len(class1))],[class1[i][1] for i in range(len(class1))],'o')
plt.plot([class2[i][0] for i in range(len(class2))],[class2[i][1] for i in range(len(class2))],'o')
plt.plot([class3[i][0] for i in range(len(class3))],[class3[i][1] for i in range(len(class3))],'o')

plt.show()