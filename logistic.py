import csv
import numpy
import math

def open_file(filename):
    f = open(filename,'r')
    f = csv.reader(f)
    X = []
    Y = []

    for lines in f:
        X.append(lines[0:256])
        Y.append(lines[256:257])
    X = numpy.matrix(X,dtype='f')
    Y = numpy.matrix(Y,dtype='f')

    #print(X[9])
    return X,Y


file_name1 = 'usps-4-9-test.csv'
file_name2 = 'usps-4-9-train.csv'

open_file(file_name1)
