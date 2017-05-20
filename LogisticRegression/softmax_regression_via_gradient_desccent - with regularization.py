"""
Handwritten digit classification

"""
import numpy
import math
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
def exp(x):
    if(x < 709):
        return math.exp(x)
    else:
        return math.exp(709)
        
def addRowToMatrix(matrix, row):
    newMatrix = numpy.zeros(shape = ((len(matrix) + 1), len(matrix[0])))
    for i in range(len(matrix)):
        newMatrix[i] = matrix[i]
    newMatrix[i+1] = row
    return newMatrix
    
def l2_norm(lamda, weights):
    return 2*lamda*weights;
    
def l1_norm(lamda, weights):
    w = numpy.ones((len(numpy.array(weights)), len(numpy.array(weights)[0])))
    for i in range(len(w)):
        for j in range(len(w[0])):
            if (weights[i,j] < 0):        
                w[i][j] = -1
    return lamda*numpy.matrix(w);

# custom sigmoid function; if -x is too large, return value 0
def sigmoid(x):
    if(-x < 709):
        return 1 / (1 + math.exp(-x))
    else:
        return 1 / (1 + math.exp(708))
    
def calculate_log_liikelihood(data, label, weights, t):
    log_likelihood = 0.0
    for j in range(0,t):
        log_likelihood += (label[j]*numpy.log(sigmoid(numpy.dot(weights, data[j])))) + ((1-label[j])*numpy.log(sigmoid(-1*numpy.dot(weights, data[j]))))
        
    return -1*log_likelihood/t
    
def plotlyGraphs(error_plot_array, labels, name):
    py1.sign_in('cgandotr', '3c9fho4498')
    trace = []
    for i in range(len(labels)):
        y1 = error_plot_array[i]
        y1 = [k for k in y1]
        #y1 = y1[:min_index[i]]
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
    if i > early_stopping_horizon:
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
            
def calculate_error(weights, data, label, k = 10):
    error = 0.0;    
    for j in range(0,len(data)):
        softmax_denom = 0.0
        softmax_num = numpy.zeros(k);
        for x in range(0,k):
            softmax_num[x] = exp(numpy.dot(numpy.transpose(weights[:,x]), data[j]))
            softmax_denom += softmax_num[x]

        prediction = numpy.argmax(softmax_num/softmax_denom)
        if(prediction != label[j]):
            error += 1;
    return error;

def softmax_loss(weights, labels, data, c):
    loss = 0.0
    for i in range(len(data)):    
        softmax_denom = 0.0
        softmax_num = numpy.zeros(c);
        for x in range(0,c):
            softmax_num[x] = exp(numpy.dot(numpy.transpose(weights[:,x]), data[i]))
            softmax_denom += softmax_num[x]
        softmax = softmax_num[labels[i]]/softmax_denom
        if (softmax > 0):
            log_softmax = math.log(softmax)
        else:
            log_softmax = 0
        loss += (log_softmax)
    return -1*loss/(len(data))
    
def getSoftmax(k, weights, training_data, j):
    softmax_denom = 0.0
    softmax_num = numpy.zeros(k);
    for x in range(0,k):
        softmax_num[x] = exp(numpy.dot(numpy.transpose(weights[:,x]), training_data[j]))
        softmax_denom += softmax_num[x]
    return softmax_num/softmax_denom

def fit(training_data, training_label, test_data, test_label, validation_data, 
        validation_label, iteration = 1000, T=2000, lamda=0.001, 
        learning_rate = 0.0001, norm = 2):
    k = len(numpy.unique(training_label))
    t = len(training_data)
    weights = numpy.matrix(numpy.zeros((len(training_data[0]), k)))
    weights_array = []
        
    early_stopping_horizon = 15    
    error_plot = numpy.zeros(iteration)
    min_error_index = 0
    loss_array = []
    loss_training = []
    loss_validation = []
    loss_testing = []
    
    accuracy_plot_array = []
    accuracy_plot_training = []
    accuracy_plot_validation = []
    accuracy_plot_testing = []    
    test_error = 0.0    
    
    for i in range(0, iteration):
        # initialise Gradient
        gradient = numpy.matrix(numpy.zeros((len(training_data[0]), k)))
        
        # calculate gradient over all the samples
        for j in range(0,t):
            modified_label = numpy.zeros(k);
            modified_label[training_label[j]] = 1
            softmax = getSoftmax(k, weights, training_data, j)
            gradient += numpy.transpose(numpy.transpose(numpy.matrix(modified_label - softmax)) * numpy.matrix(training_data[j]))
        
        norm_term = 0.0
        if (norm == 2):
            norm_term = l2_norm(lamda, weights)
        else:
            norm_term = l1_norm(lamda, weights)
        # update weights vector according to the update rule of Gradient descent method
        weights = weights + learning_rate * (gradient  - norm_term)
        
        loss_training.append(softmax_loss(weights, training_label, training_data, k))        
        loss_validation.append(softmax_loss(weights, validation_label, validation_data, k))
        loss_testing.append(softmax_loss(weights, test_label, test_data, k))
        
        # calculating error percentage on train, test and validation data
        training_error = calculate_error(weights, training_data, training_label)
        validation_error = calculate_error(weights, validation_data, validation_label)
        test_error = calculate_error(weights, test_data, test_label)
        
        accuracy_plot_training.append((len(training_data) - training_error)*100/len(training_data))
        accuracy_plot_validation.append((len(validation_data) - validation_error)*100/len(validation_data))
        accuracy_plot_testing.append((len(test_data) - test_error)*100/len(test_data))
        
        learning_rate = learning_rate/(1+i/T)
        
        error_plot[i] = validation_error*100/len(validation_data);
        weights_array.append(weights)        

        # check for early stopping
        if (early_stopping(early_stopping_horizon, accuracy_plot_validation, i)):
            min_error_index = i-early_stopping_horizon;
            weights = weights_array[min_error_index]            
            break
        min_error_index = i
        
    loss_array.append(loss_training)
    loss_array.append(loss_validation)
    loss_array.append(loss_testing)
        
    accuracy_plot_array.append(accuracy_plot_training)
    accuracy_plot_array.append(accuracy_plot_validation)
    accuracy_plot_array.append(accuracy_plot_testing)
    
    # printing error on training and testing dataset
    print('Error on validation dataset : ' + str(error_plot[min_error_index]) + '%');
    print('Error on test dataset : ' + str(test_error*100/len(test_data)) + '%');

    return weights, loss_array, accuracy_plot_array
    
#---------------------------------------Main function------------------------------------------------------

if __name__ == "__main__":        
    numpy.random.seed(0)
    learning_rate = 0.0001
    N = 20000
    N_test = 2000
    lamda = 0.001       # regularization weightage parameter
    T = 2000
    iteration = 1000
    training_data, training_label, test_data, test_label, validation_data, validation_label = get_data(N, N_test)
    
    weights, loss_array, accuracy_plot_array = fit(training_data, training_label, 
                                                    test_data, test_label, validation_data, 
                                                    validation_label)    
    
    plotlyGraphs(loss_array, ['Training Set','Validation Set','Test Set'], "Loss Function and Iterations")
    plotlyGraphs(accuracy_plot_array, ['Training Set','Validation Set','Test Set'], "Accuracy and Iterations")
    