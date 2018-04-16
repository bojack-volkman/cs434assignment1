import numpy
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
from numpy.linalg import inv #for inverse
from numpy import matrix
#import matplotlib.pyplot as plt

def open_file(filename,check_dummy): #check_dummy == 1: dummy on
	f = open(filename,'r')
	X = []
	Y = []

	for line in f:
		lines = line.split()
		for a in range(len(lines)):
			lines[a] = float(lines[a])
		if check_dummy == 1:
			X.append([1] + lines[0:13]) #dummy as 1
		else:
			X.append(lines[0:13])
		Y.append(lines[13])

	return X, Y

def cal_weight(X,Y):
	X_tr = numpy.transpose(X)
	Y_tr = numpy.transpose(Y)
	#https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
	w = numpy.dot(inv(numpy.dot(X_tr,X)),numpy.dot(X_tr,Y))

	return w

def ASE(X,Y,w):
	temp = 0
	for a in range (len(Y)):
		temp += (float((Y[a] - numpy.dot(X[a],w))))**2

	#print("length of Y = " + str(len(Y)))
	return temp/len(Y) #this is ASE.
	#return temp

file_name1 = 'housing_train.txt'
file_name2 = 'housing_test.txt'

#question 1.1
tr_X, tr_Y = open_file(file_name1,1) #transposes training data with dummy var
ts_X, ts_Y = open_file(file_name2,1) #transposes testing data with dummy var

w = cal_weight(tr_X,tr_Y) #transposes into learned weight vector (creates model)

print('1.1 Weight with dummy:')
print(numpy.matrix(w)) #answer to 1.1

#print(tr_X)

#question 1.2
print('1.2 ASE with dummy:')
print('Traning data: ' + str(ASE(tr_X, tr_Y,w))) #prints ASE of training data
print('Testing data: ' + str(ASE(ts_X, ts_Y,w))) #prints ASE of testing data
#print(matrix(tr_X).shape)

#question 1.3
tr_X, tr_Y = open_file(file_name1,0) #transposes training data
ts_X, ts_Y = open_file(file_name2,0) #transposes testing data

w = cal_weight(tr_X,tr_Y) #transposes into learned weight vector (creates model)
print('Weight without dummy:')
#print(tr_X)
print(numpy.matrix(w)) ##probably don't need this line anymore?

print('ASE without dummy:') 
print('Traning data: ' + str(ASE(tr_X, tr_Y,w))) #prints ASE of training data
print('Testing data: ' + str(ASE(ts_X, ts_Y,w)))

#question 1.4

rand_feature = 0
temp_tr_X = tr_X
temp_ts_X = ts_X

for a in range(2,100): #choose number
	temp1 = numpy.random.normal(0,1.0,(len(temp_tr_X))) # random normal standard distribution for training value
	temp2 = numpy.random.normal(0,1.0,(len(temp_ts_X))) # random normal standard distribution for testing value
	#Translates slice objects to concatenation along the second axis.
	temp_tr_X = numpy.c_[temp_tr_X,temp1]
	temp_ts_X = numpy.c_[temp_ts_X,temp2]
	#print(matrix(temp_tr_X).shape)
	if (a % 2 == 0):
		print("d = " + str(a))
		w = cal_weight(temp_tr_X,tr_Y)
		print("Size of matrix: " + str(matrix(temp_tr_X).shape))
		print('ASE when d=' + str(a))
		print("traning data ASE: "+ str(ASE(temp_tr_X,tr_Y,w)))
		print("testing data ASE: "+ str(ASE(temp_ts_X,ts_Y,w)))

		#plt.plot(a,ASE(temp_tr_X,tr_Y,w),'bo')
		#plt.plot(a,ASE(temp_ts_X,ts_Y,w),'go')

#plt.show()

#print(temp1)
#print(matrix(temp_tr_X).shape)
#print(matrix(temp1).shape)
