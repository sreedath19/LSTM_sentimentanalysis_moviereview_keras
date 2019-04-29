import numpy as np
import random as r
ids = np.load('idmatmagic.npy')



labels = []
for i in range (0,24999):
	if i < 12499:
		labels.append(1)
	else:
		labels.append(0)


index = []
train_y = []
r.seed(1)
for i in range (0,22999):
	j = r.randint(0,24999)
	index.append(j)
	train_y.append(labels[j])
	
print(index)
print('************************************************************************************************************************************************************************************************************************************************************************')
negind = []
test_y = []
for i in range (0,24999):
	if i not in index:
		negind.append(i)
		test_y.append(labels[i])
print(negind)
print('***********************************************************************************************************************************************************************************************************************************************************************8')

train_x = ids[index]
test_x = ids[negind]




np.save('train_x',train_x)
np.save('train_y',train_y)
np.save('test_x',test_x)
np.save('test_y',test_y)



np.savetxt('train_x.csv', train_x, "%d", "")
np.savetxt('train_y.csv', train_y, "%d", "")
np.savetxt('test_x.csv', test_x, "%d", "")
np.savetxt('test_y.csv', test_y, "%d", "")

#trainx = np.load('train_x.npy')
#print(type(train_x))
