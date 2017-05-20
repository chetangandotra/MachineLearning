"""
MNIST - Handwritten digit classification
"""
import numpy
import math
import matplotlib.pylab as py
import matplotlib.pyplot as plt
import plotly.plotly as py1
import plotly.graph_objs as go

from LoadMNIST import load_mnist

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
    training_data = addOnesColAtStart(training_data)
    test_data = addOnesColAtStart(test_data)
    
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
    
# custom sigmoid function; if -x is too large, return value 0
def sigmoid(x):
    if(-x < 700):
        return 1 / (1 + math.exp(-x))
    else:
        return 0.0

def extract_digit_specific_data(digits, data, label):
    pruned_data = numpy.zeros(shape = (1, len(data[0])))
    pruned_labels = []
    cnt = 0
    for i in range(0, len(label)):
        if label[i] in digits:
            if (cnt == 0):
                for j in range(len(data[0])):
                    pruned_data[0][j] = data[i][j]
            else:
                pruned_data = addRowToMatrix(pruned_data, data[i])
            pruned_labels.append(digits.get(label[i]))
            cnt = cnt + 1
    return pruned_data, pruned_labels
        
def addRowToMatrix(matrix, row):
    newMatrix = numpy.zeros(shape = ((len(matrix) + 1), len(matrix[0])))
    for i in range(len(matrix)):
        newMatrix[i] = matrix[i]
    newMatrix[i+1] = row
    return newMatrix
    
def l2_norm(lamda, weights):
    return 2*lamda*weights;
    
def l1_norm(lamda, weights):
    w = numpy.ones(len(weights))
    for i in range(len(weights)):
        if (weights[i] < 0):        
            w[i] = -1
    return lamda*w;
    
# 5.b, 5.c - Plot of Percent correct on training data v/s iterations, 
    #length of weight vector v/s lambda
def plotlyGraphs(error_plot_array, lamda_vals, min_index = -1):
    py1.sign_in('chetang', 'vil7vTAuCSWt2lEZvaH9')

    trace = []
    
    for i in range(len(lamda_vals)):
        y1 = error_plot_array[i]
        #y1 = y1[:min_index[i]]
        x1 = [j+1 for j in range(len(y1))]
        
        trace1 = go.Scatter(
            x=x1,
            y=y1,
            name = 'lambda = ' + (str)(lamda_vals[i]), # Style name/legend entry with html tags
            connectgaps=True
        )

        trace.append(trace1)
    print (len(trace))
    data = trace

    fig = dict(data=data)
    py1.iplot(fig, filename='percentCorrect-connectgaps_1')
    
# 5.d - Final test error v/s Lambda values graph
def plotlyErrorVsLamda(test_error_array, lamda_vals):
    lamda_vals1 = [math.log(lamda) for lamda in lamda_vals]
    py1.sign_in('chetang', 'vil7vTAuCSWt2lEZvaH9')
    trace = []
    colors = ['rgb(104,224,204)', 'rgb(204,204,204)', 'rgb(49,130,189)', 'rgb(41,180,129)']    
    
    for i in range(0, len(test_error_array)):
        y1 = test_error_array[i]
        x1 = lamda_vals1[i]
        
        trace1 = go.Bar(
            x=x1,
            y=y1,
            name='Lambda = log(' + (str)(lamda_vals[i]) + ')',
            marker=dict(
            color=colors[i]
            )
        )
        trace.append(trace1)
    
    layout = go.Layout(
        xaxis=dict(tickangle=-45),
        barmode='group',
    )

    fig = go.Figure(data=trace, layout=layout)
    py1.iplot(fig, filename='angled-text-bar')

def dropFirstColumns(weights):
    return numpy.array(weights)[0][1:]

def dropFirstVal(weights):
    return weights[1:]
    
#---------------------------------------Main function------------------------------------------------------
        
if __name__ == "__main__":        
    numpy.random.seed(0)
    learning_rate = 0.0001
    N = 20000
    N_test = 2000
    lamda_vals = [0.1]#, 100, 10, 1] # regularization weightage parameter
    T = 2000
    iteration = 200
    
    training_data, training_label, test_data, test_label, validation_data, validation_label = get_data(N, N_test)
 
    # Training on 2's and 3's dataset 
    digits = {2:1, 8:0}
    training_data, training_label = extract_digit_specific_data(digits, training_data, training_label)
    validation_data, validation_label = extract_digit_specific_data(digits, validation_data, validation_label)
    test_data, test_label = extract_digit_specific_data(digits, test_data, test_label)
    
    t = len(training_data)
    t_validation = len(validation_data)

    error_plot_array = []
    percent_correct_plot_array = []
    weight_vector_length_array = []
    test_error_array = []
    min_index = []
    
    for lamda in lamda_vals:
    
        #weights = numpy.matrix(numpy.zeros(len(training_data[0])))
        weights = numpy.zeros(len(training_data[0]))
        weights_array = []
        error_plot = []
        percent_correct_plot = []
        weight_vector_length = []
        min_error_index = 0
        learning_rate = 0.0001
    
        for i in range(0,iteration):
            # initialise Gradient
            #gradient = numpy.matrix(numpy.zeros(len(training_data[0])))
            gradient = numpy.zeros(len(training_data[0]))
            
            # calculate gradient over all the samples
            for j in range(0,t):
                gradient += ((training_label[j]) - sigmoid(numpy.dot(weights, training_data[j])))*numpy.array(training_data[j])
                                    
            # update weights vector according to the update rule of Gradient descent method
            weight_vector_length.append(numpy.linalg.norm(weights))
            weights = weights + learning_rate * (gradient - l1_norm(lamda, weights))
        
            learning_rate = learning_rate/(1+i/T)
        
            # calculating accuracy percentage on training data
            percent_correct = 0.0;    
            for j in range(0,len(training_data)):
                prediction = sigmoid(numpy.dot(weights, training_data[j]))
                if(prediction>0.5 and training_label[j]==1):
                    percent_correct += 1;
                elif (prediction<=0.5 and training_label[j]==0):
                    percent_correct += 1;            
            
            # calculating error percentage on validation data
            error = 0.0;    
            for j in range(0,len(validation_data)):
                prediction = sigmoid(numpy.dot(weights, validation_data[j]))
                if(prediction>0.5 and validation_label[j]!=1):
                    error += 1;
                elif (prediction<=0.5 and validation_label[j]!=0):
                    error += 1;
                
            error_plot.append(error*100/t_validation);
            percent_correct_plot.append(percent_correct*100/t)
            weights_array.append(weights)        

            # check for early stopping
            early_stopping_horizon = 20
            if i > early_stopping_horizon:
                counter = 0;
                if (error_plot[i] > error_plot[i-1] and error_plot[i-1] > error_plot[i-2]):
                    min_error_index = i-2
                    weights = weights_array[i-2]
                    break
                for p in range(0,early_stopping_horizon):
                    if error_plot[i-p] >= error_plot[i-p-1]:
                        counter+=1
                    else:
                        break
            
                if counter == early_stopping_horizon:
                    min_error_index = i-early_stopping_horizon;
                    weights = weights_array[min_error_index]
                    break
 
            min_error_index = i
            #print(error_plot[i])
    
        error_plot_array.append(error_plot)
        percent_correct_plot_array.append(percent_correct_plot)
        min_index.append(min_error_index)
        weight_vector_length_array.append(weight_vector_length)
        
        error = 0.0;    
        for j in range(0,len(test_data)):
            prediction = sigmoid(numpy.dot(weights, test_data[j]))
            if(prediction>0.5 and test_label[j]!=1):
                error += 1;
            elif (prediction<=0.5 and test_label[j]!=0):
                error += 1;
    
        # printing error on training and testing dataset
        print ('lambda = ' + (str)(lamda))
        print('Error on validation dataset : ' + str(error_plot[min_error_index]) + '%');
        print('Error on test dataset : ' + str(error*100/len(test_data)) + '%');
        test_error_array.append(error*100/len(test_data))

    #plotlyGraphs(error_plot_array, lamda_vals, min_index)
    #plotlyGraphs(percent_correct_plot_array, lamda_vals)
    #plotlyGraphs(weight_vector_length_array, lamda_vals)
    #plotlyErrorVsLamda(test_error_array, lamda_vals)
    # The rest of columns are pixels
    pixels1 = dropFirstVal(weights)

    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = numpy.reshape(pixels1, (28, 28))

    # Reshape the array into 28 x 28 array (2-dimensional array)
    #pixels = pixels.reshape((28, 28))
    
    plt.imshow(pixels, cmap='gray')
    plt.show()

    