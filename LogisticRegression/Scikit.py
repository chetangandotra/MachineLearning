# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 10:36:04 2017

@author: Chetan
"""

import numpy
from LoadMNIST import load_mnist
from sklearn import linear_model

#----------------------------------------Utility functions--------------------------------------
def get_data(N, N_test):
    #load MNIST data using libraries available
    training_data, training_labels = load_mnist('training')    
    test_data, test_labels = load_mnist('testing')
    
    training_data = flatArray(N, 784, training_data) #training_data is N x 784 matrix
    training_labels = training_labels[:N]
    test_data = flatArray(N_test, 784, test_data)
    test_labels = test_labels[:N_test]

    # adding column of 1s for bias
    #training_data = addOnesColAtStart(training_data)
    #test_data = addOnesColAtStart(test_data)
    
    # Last 10% of training data size will be considered as the validation set
    N_validation = int (N / 10)
    validation_data = training_data[N-N_validation:N]
    validation_labels = training_labels[N-N_validation:N]
    N=N-N_validation
    #update training data to remove validation data
    training_data = training_data[:N]
    training_labels = training_labels[:N]    

    return training_data, training_labels, test_data, test_labels, validation_data, validation_labels
    
def flatArray(rows, cols, twoDArr):
    flattened_arr = numpy.zeros(shape=(rows, cols))
    for row in range(0, rows):
        i=0
        for element in twoDArr[row]:
            for el1 in element:
                flattened_arr[row][i] = el1            
                i = i+1
    return flattened_arr
    
def addOnesColAtStart(matrix):
    Ones = numpy.ones(len(matrix))
    newMatrix = numpy.c_[Ones, matrix]
    return newMatrix

#---------------------------------------Main function------------------------------------------------------
        
if __name__ == "__main__":        
    numpy.random.seed(0)
    learning_rate = 0.0001
    N = 20000
    N_test = 2000
    lamda = 0.0001        # regularization weightage parameter
    T = 2000
    iteration = 150
    training_data, training_label, test_data, test_label, validation_data, validation_label = get_data(N, N_test)
    k = len(numpy.unique(training_label))
    
    print (len(training_data), len(training_label))
    print (len(validation_data), len(validation_label))
    
    lr = linear_model.LogisticRegression(n_jobs=-1, solver='lbfgs')
    lr.fit(training_data, training_label)
    predictedVals = lr.predict(test_data[0:N_test]) 
    accuracy= numpy.sum(predictedVals==test_label[0:N_test])
    print (accuracy)
    
    
    