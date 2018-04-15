import numpy
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
from numpy.linalg import inv #for inverse
 
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
	return temp/len(Y) #this is ASE.

def add_feature(rand_num):
	
	#while index is greater than 0
		#some variable d = random sample from normal distribution
		#append d to instance of X
		#index--
		
		#however, I have no idea how to implement it in this version of Python
	
	#Working on it now
	return

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
print(ASE(tr_X, tr_Y,w)) #prints ASE of training data
print(ASE(ts_X, ts_Y,w)) #prints ASE of testing data

#question 1.3
tr_X, tr_Y = open_file(file_name1,0) #transposes training data
ts_X, ts_Y = open_file(file_name2,0) #transposes testing data

w = cal_weight(tr_X,tr_Y) #transposes into learned weight vector (creates model)
print('Weight without dummy:')
#print(tr_X)
print(numpy.matrix(w)) ##probably don't need this line anymore?

print('ASE without dummy:') 
print(ASE(tr_X, tr_Y,w)) #prints ASE of training data
print(ASE(ts_X, ts_Y,w)) #prints ASE of testing data

#question 1.4