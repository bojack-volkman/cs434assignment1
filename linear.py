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
	return temp/len(Y)

def add_feature(rand_num):
	#Working on it now
	return

file_name1 = 'housing_train.txt'
file_name2 = 'housing_test.txt'

tr_X, tr_Y = open_file(file_name1,1) # for question 1
ts_X, ts_Y = open_file(file_name2,1)
w = cal_weight(tr_X,tr_Y)
print('Weight with dummy:')
print(numpy.matrix(w)) #question 1
#print(tr_X)
print('ASE with dummy:')
print(ASE(tr_X, tr_Y,w))
print(ASE(ts_X, ts_Y,w))

tr_X, tr_Y = open_file(file_name1,0) # w/o dummy
ts_X, ts_Y = open_file(file_name2,0)
w = cal_weight(tr_X,tr_Y)
print('Weight without dummy:')
#print(tr_X)
print(numpy.matrix(w))
print('ASE without dummy:') 
print(ASE(tr_X, tr_Y,w))
print(ASE(ts_X, ts_Y,w))