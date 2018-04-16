import csv
import numpy
import math

#print(numpy.exp(range(1,5))) #exp fucntion

def open_file(filename):
    f = open(filename,'r')
    f = csv.reader(f)
    X = []
    Y = []

    for lines in f:
        X.append(lines[0:256])
        Y.append(lines[256:257])
    X = numpy.array(X,dtype='f')
    Y = numpy.array(Y,dtype='f')

    #print(X[9])
    #Prevent overflow
    return X * (1.0/255.0),Y
    #return X,Y

file_name1 = 'usps-4-9-test.csv'
file_name2 = 'usps-4-9-train.csv'

ts_X,ts_Y  = open_file(file_name1)
tr_X,tr_Y = open_file(file_name2)

w = numpy.zeros(256)
learning = 0.005 #Learning Rate
count = 0
temp = []
#print(numpy.exp(range(1,5))) #exp fucntion
#print(numpy.finfo(float).eps) #epsilon

#Algorithm is from our slides
#https://oregonstate.instructure.com/courses/1674445/files/folder/lectures?preview=70749128
acc_pass = 0
acc_total = 0

print('python is iterating now......')
while count < 1000: # number of iteration
    delta = numpy.zeros(256)
    for i in range(0,(numpy.matrix(ts_X).shape[0])):
        y_hat = float(1) / float(1 + numpy.exp(-1*numpy.dot(numpy.transpose(w),tr_X[i])))
        if y_hat >= 0.5:
            y_hat = 1
            if ts_Y[i] == 1:
                acc_pass = acc_pass+1
        else:
            y_hat = 0
            if ts_Y[i] == 0:
                acc_pass = acc_pass+1               
        acc_total = acc_total+1
        new_delta = ((y_hat - tr_Y[i]) * tr_X[i])
        #print(new_delta)
        delta = delta + new_delta
    w = w - (learning*delta)
    #if (numpy.linalg.norm(delta)) <= numpy.finfo(float).eps:
    #if count == 1:
    #    print("Over")
    #   break;
    count = count + 1
    #print(count)

print("Success percentage : " + str(float(acc_pass)/float(acc_total)))
print("How many iteration? :" + str(count))
#print(w)