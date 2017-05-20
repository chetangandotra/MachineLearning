"""
CSE 253: Neural Networks and Pattern Recognition
Logistic Regression With and Without Gradient Descent
This file contains code for questions 4 and 5, including all graph plots
"""
import numpy
import math
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
    if(-x < 709):
        return 1 / (1 + math.exp(-x))
    else:
        return 1 / (1 + math.exp(708))

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
    
def calculate_log_liikelihood(data, label, weights, t):
    log_likelihood = 0.0
    for j in range(0,t):
        #print(numpy.log(sigmoid(numpy.dot(weights, data[j]))))
        log_likelihood += (label[j]*numpy.log(sigmoid(numpy.dot(weights, data[j])))) + ((1-label[j])*numpy.log(sigmoid(-1*numpy.dot(weights, data[j]))))
        
    return -1*log_likelihood/t

# 5.b, 5.c - Plot of Percent correct on training data v/s iterations, 
#length of weight vector v/s lambda
def plotlyGraphsRegularization(error_plot_array, lamda_vals, graph_name, min_index = -1):
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
    data = trace

    fig = dict(data=data)
    py1.iplot(fig, filename=graph_name)
    
def plotlyGraphs(error_plot_array, labels, name):
    py1.sign_in('cgandotr', '3c9fho4498')
    trace = []
    
    for i in range(len(labels)):
        y1 = error_plot_array[i]
        y1 = [k for k in y1]
        x1 = [(j+1) for j in range(len(y1))]
        
        trace1 = go.Scatter(
            x=x1,
            y=y1,
            name = str(labels[i]), # Style name/legend entry with html tags
            connectgaps=True
        )
        trace.append(trace1)
    data = trace
    fig = dict(data=data)
    py1.iplot(fig, filename=name)
    
def early_stopping(early_stopping_horizon, accuracy, i):
    if i > early_stopping_horizon + 1:
        counter = 0;
        for p in range(0,early_stopping_horizon):
            if accuracy[i-p] <= accuracy[i-p-1]:
                counter+=1
            else:
                break
        
        if(counter == early_stopping_horizon):
            return True
        else:
            return False
            
def calculate_error(weights, data, label):
    error = 0.0;    
    for j in range(0,len(data)):
        prediction = sigmoid(numpy.dot(weights, data[j]))
        if(prediction>0.5 and label[j]!=1):
            error += 1;
        elif (prediction<=0.5 and label[j]!=0):
            error += 1;
    return error;
    
def dropFirstColumn(weights):
    return numpy.array(weights)[0][1:]

def fit(training_data, training_label, test_data, test_label, validation_data, 
        validation_label, digits, learning_rate=0.0001, iteration=200, batch_size=0, T=5000, lamda_vals=[0], norm=2):
    
    t = len(training_data)
    accuracy_plot_array = []
    log_likelihood_array = []
    weight_vector_length_array = []
    accuracy_plot_training_array = []
    weights_for_all_lamda = []
    test_error_array = []
    org_learning_rate = learning_rate
    
    for lamda in lamda_vals:
        weights = numpy.matrix(numpy.zeros(len(training_data[0])))
        weights_array = []
        weight_vector_length = []
        
        accuracy_plot_training = []
        accuracy_plot_validation = []
        accuracy_plot_testing = []
        
        log_likelihood_training = []
        log_likelihood_validation = []
        log_likelihood_testing = []
        min_error_index = 0
        learning_rate = org_learning_rate
    
        for i in range(0, iteration):
            # initialise Gradient
            gradient = numpy.matrix(numpy.zeros(len(training_data[0])))
            cnt = 0
        
            norm_term = []
            if (norm == 2):
                norm_term = l2_norm(lamda, weights)
            else:
                norm_term = l1_norm(lamda, weights)
                
            # calculate gradient over all the samples
            for j in range(0,t):
                # update gradient
                gradient += ((training_label[j]) - sigmoid(numpy.dot(weights, training_data[j])))*numpy.array(training_data[j])
                cnt += 1            
                if (batch_size != 0 and cnt == (int)(t/batch_size)):
                    cnt = 0
                    weight_vector_length.append(numpy.linalg.norm(weights))
                    weights = weights + learning_rate * (gradient - norm_term)
                    # re-initialise Gradient
                    gradient = numpy.matrix(numpy.zeros(len(training_data[0])))
                    # calculating log likelhood of training, validation and test dataset
                    log_likelihood_training.append(calculate_log_liikelihood(training_data, training_label, weights, t))
                    log_likelihood_validation.append(calculate_log_liikelihood(validation_data, validation_label, weights, len(validation_data)))
                    log_likelihood_testing.append(calculate_log_liikelihood(test_data, test_label, weights, len(test_data)))
        
            if (batch_size == 0):
                weight_vector_length.append(numpy.linalg.norm(weights))
                # update weights vector according to the update rule of Gradient descent method
                weights = weights + learning_rate * (gradient - norm_term)
                
                # calculating log likelhood of training, validation and test dataset
            log_likelihood_training.append(calculate_log_liikelihood(training_data, training_label, weights, t))
            log_likelihood_validation.append(calculate_log_liikelihood(validation_data, validation_label, weights, len(validation_data)))
            log_likelihood_testing.append(calculate_log_liikelihood(test_data, test_label, weights, len(test_data)))
                
            # anneling of learning rate
            learning_rate = learning_rate/(1+i/T)
        
            # calculating error percentage on train, test and validation data
            accuracy_plot_training.append((len(training_data) - calculate_error(weights, training_data, training_label))*100/len(training_data))
            accuracy_plot_validation.append((len(validation_data) - calculate_error(weights, validation_data, validation_label))*100/len(validation_data))
            accuracy_plot_testing.append((len(test_data) - calculate_error(weights, test_data, test_label))*100/len(test_data))

            weights_array.append(weights)        

            # check for early stopping
            early_stopping_horizon = 15
            min_error_index = i
            if(early_stopping(early_stopping_horizon, accuracy_plot_validation, i) and i > early_stopping_horizon):
                min_error_index = i-early_stopping_horizon;
                weights = weights_array[min_error_index]
                break
           
        weight_vector_length_array.append(weight_vector_length)
        weights_for_all_lamda.append(weights)
        
        log_likelihood_array.append(log_likelihood_training);
        log_likelihood_array.append(log_likelihood_validation);
        log_likelihood_array.append(log_likelihood_testing);
        accuracy_plot_array.append(accuracy_plot_training)
        accuracy_plot_array.append(accuracy_plot_validation)
        accuracy_plot_array.append(accuracy_plot_testing)  
        accuracy_plot_training_array.append(accuracy_plot_training)
        
        test_error = (calculate_error(weights, test_data, test_label)*100)/len(test_data)
        validation_error = (calculate_error(weights, validation_data, validation_label)*100)/len(validation_data)
        print('Error on validation dataset : ' + str(validation_error) + '%');
        print('Error on test dataset : ' + str(test_error) + '%');
        test_error_array.append(test_error)
    
    #For single lambda value
    if (len(lamda_vals) == 1):
        plotlyGraphs(log_likelihood_array, ['Training Set','Validation Set','Test Set'], 'Log Likelihood Plot')    
        plotlyGraphs(accuracy_plot_array, ['Training Set','Validation Set','Test Set'], 'Percent Correct Plot')    
    #else:
        #For multiple lambda values
        #plotlyGraphsRegularization(accuracy_plot_training_array, lamda_vals, "Accuracy vs Epoch")
        #plotlyGraphsRegularization(weight_vector_length_array, lamda_vals, "Weight Vector Length vs Epoch")
        #plotlyErrorVsLamda(test_error_array, lamda_vals)
    
    # printing error on training and testing dataset
    return weights_for_all_lamda
    
def plot_weights(weights):
    lamda_vals = [1000, 100, 10, 1, 0.1, 0.001, 0.0001]
    i = 0
    for w in weights:
        # Plot weights as image after removing bias terms. The rest of columns are pixels
        print ('lambda = ' + str(lamda_vals[i]))
        i += 1
        pixels1 = dropFirstColumn(w)
        pixels = numpy.reshape(pixels1, (28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()    
    
def l2_norm(lamda, weights):
    return 2*lamda*weights;
    
def l1_norm(lamda, weights):
    w = numpy.ones(len(weights))
    for i in range(len(weights)):
        if (weights[i] < 0):        
            w[i] = -1
    return lamda*w;
    
# 5.d - Final test error v/s Lambda values graph
# Generates bar graphs
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
    py1.iplot(fig, filename='Final Error vs Lambda')
    
#---------------------------------------Main function------------------------------------------------------
        
if __name__ == "__main__":        
    numpy.random.seed(0)
    
    N = 20000
    N_test = 2000
    iteration = 200
    batch_size = 0
    T = 2000    
    
    #lamda_vals = [0]
    lamda_vals = [1000, 100, 10, 1, 0.1, 0.001, 0.0001] # regularization weightage parameter. Put [0] for no regularization
    norm = 2 # 2 for l-2 norm, 1 for l-1 norm
    
    full_training_data, full_training_label, full_test_data, full_test_label, full_validation_data, full_validation_label = get_data(N, N_test)
 
    # Parameters for training data on 2's and 3's
    digits = {2:1, 3:0}
    training_data, training_label = extract_digit_specific_data(digits, full_training_data, full_training_label)
    validation_data, validation_label = extract_digit_specific_data(digits, full_validation_data, full_validation_label)
    test_data, test_label = extract_digit_specific_data(digits, full_test_data, full_test_label)
    learning_rate = 0.0001 
    
    weights23 = fit(training_data, training_label, test_data, test_label, 
                    validation_data, validation_label, learning_rate, iteration, 
                    batch_size, digits, T, lamda_vals, norm)
    plot_weights(weights23)

    # Parameters for training data on 2's and 8's
    digits = {2:1, 8:0}
    training_data, training_label = extract_digit_specific_data(digits, full_training_data, full_training_label)
    validation_data, validation_label = extract_digit_specific_data(digits, full_validation_data, full_validation_label)
    test_data, test_label = extract_digit_specific_data(digits, full_test_data, full_test_label)
    learning_rate = 0.1 
    lamda_vals = [0] #Regularization not asked for with 2/8 case

    weights28 = fit(training_data, training_label, test_data, test_label, 
                    validation_data, validation_label, digits, learning_rate, iteration, 
                    batch_size, T, lamda_vals, norm)
    plot_weights(weights28)
    
    weights = weights28[0] - weights23[0]

    plot_weights(weights)
    