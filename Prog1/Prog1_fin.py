#Cera Oh. CS545 ML. Prog 1. Winter 2022
import multiprocessing as mp 
import numpy as np
from numpy.typing import _128Bit
import pandas as pd
from matplotlib import pyplot as plt
from array import array

#Data Path
train_path = './mnist_train.csv'
test_path = './mnist_test.csv'

#Function that loads csv data files
def csv_load(file_path):
    with open(file_path, 'r') as csv_file:
        dt = np.genfromtxt(file_path, delimiter=',',skip_header=1)
    preprocess_dt, answer = preprocess(dt)

    return preprocess_dt, answer

#Function to preprocess data
def preprocess(data):
    answer = data[:, 0] #save labels as its own vector
    values = data[:,1:]/255 #remove labels, scale data
   
    answer = answer.reshape(len(answer), 1) #reshape matrix for easier calc later on

    ones = np.ones(np.shape(answer)) #create a new matrix filled with ones to serve as bias input node
    preprocess_data = np.hstack((ones, values)) #added bias node to input data

    return preprocess_data, answer


#Function that initializes the weights for the input x neuron weight matrix and change matrix to keep track of changes in weights
def initialize_weights(m, n):
    weights = np.random.randint(-5, 5, size=(m, n))/100 #set weights to random number between -0.05 to 0.05
    change = np.zeros((m,n)) #makes change in weight matrix filled with zeros

    return weights, change  


#Function for forward propagation that returns sigma 
def forwards(inputs, weights):
    dot_pro = np.dot(inputs, weights) #calculates dot product of the current neuron weights and inputs. 
    z = -1 * dot_pro
    sigma = 1/(1 + np.exp(z))#calc sigma

    return sigma

#Function for backwards propagation
def backwards(hidden, output, answer, h_weights, o_weights, row, n, learn, momentum, change_o, change_h):
    o_delta = np.zeros((1, 10)) #array to keep track of outer layer's delta values
    h_delta = np.zeros((1, int(n))) #array to keep track of hidden layer's delta values

    #reshape arrays for easier calc 
    output = output.reshape(1, 10)
    hidden = hidden.reshape(1, int(n) + 1)
    row = row.reshape(1, 785)

    #calc delta for output layer
    o_delta = output * (1 - output) * (answer - output)

    #calc delta for input layer
    dot = np.dot(o_weights, o_delta.T).T
    h_delta = hidden * (1 - hidden) * dot
  
    #update weights for output layer
    prev_o = np.copy(o_weights)
    o_weights += (learn * hidden.T * o_delta) + (momentum * change_o)   
    change_o = o_weights - prev_o

    #update weights for hidden layer
    prev = np.copy(h_weights)
    h_weights += (learn * np.tile(h_delta[0, 1:], (785, 1)) * np.tile(row.T, (1,n))) + (momentum * change_h)
    change_h = h_weights - prev

    return h_weights, o_weights, change_o, change_h

#Function for plotting and initializing weight and change matrices for runs
def run_epoch_pal(n, momentum, learn, train_data, train_ans, test_data, test_ans):
    num = n + 1
    num2 = n
    h_weights, change_h = initialize_weights(785, num2) #initialize hidden layer weight matrix and change matrix
    o_weights, change_o = initialize_weights(num, 10) #initialize output layer weight matrix and change matrix

    #Call run_epoch, gather results for printing
    plt_train, plt_test, c_matrix = run_epoch(num2, h_weights, o_weights, change_o, change_h, train_ans, test_ans, train_data, test_data, learn, momentum)
    
    print(plt_train)
    print(plt_test)
    print(c_matrix)

    #Plots graphs and saves the graph
    plt_data = pd.DataFrame([plt_test, plt_train]).T
    plt_data.columns = ['Test', 'Train']
    for col in plt_data.columns:
        plt.plot(plt_data.index, plt_data[col], label=col)
    plt.legend()
    plt.title('Accuracy: N {} | Momentum {} | Size {}'.format(n, momentum, len(train_ans)))
    plt.savefig('AccuracyGraph_N{}_Momentum{}_Train N{}.png'.format(n, momentum, len(train_ans)))

    return (plt_train, plt_test, c_matrix)

#Function calculates accuracy for the 0th Epoch
def calc_zero_accuracy(n, h_weights, o_weights, ans, data):
    tp, fp = 0, 0
    i = 0

    for row in data:#per row of sample      
        prediction = -1 #keeps track of which output neuron fired
        output = np.repeat(0, 10).reshape(1, 10) #array to keep track of outer layer's sigma values
        hidden = np.ones((1, n-1)) #array to keep track of hidden layer's sigma values
        answer = np.repeat(0.1, 10).reshape(1, 10) #array for target value, filled with 0.1
   
        expected = int(ans[i][0]) #check answer        
        answer[0][expected] = 0.9 #update the correct target value as 0.9

        #Forward propagation
        hidden = forwards(row, h_weights)
        hidden = np.insert(hidden, 0, 1) #add bias node to hidden layer
        output = forwards(hidden, o_weights)

        prediction = max(output) #get prediction

        #For Acccuracy 
        o_index = np.where(output==prediction)[0][0] 
        if expected == o_index:
            tp += 1
        else:
            fp += 1

        i += 1

    #calc accuracy for the epoch
    accuracy = calc_accuracy(tp, fp)

    return accuracy 

#Function calculates accuracy
def calc_accuracy(tp, fp):
    accuracy = (tp)/(tp + fp)
    return accuracy 

#main NN function
def neural_network(n, h_weights, o_weights, change_o, change_h, ans, data, learn, momentum, c_matrix = None):
    tp, fp = 0, 0
    i = 0

    for row in data:#per row of sample      
        prediction = -1 #keeps track of which output neuron fired
        output = np.repeat(0, 10).reshape(1, 10) #array to keep track of outer layer's sigma values
        hidden = np.ones((1, n-1)) #array to keep track of hidden layer's sigma values
        answer = np.repeat(0.1, 10).reshape(1, 10) #array for target value, filled with 0.1
   
        expected = int(ans[i][0]) #check answer        
        answer[0][expected] = 0.9 #update the correct target value as 0.9

        #Forward propagation
        hidden = forwards(row, h_weights)
        hidden = np.insert(hidden, 0, 1) #add bias node to hidden layer
        output = forwards(hidden, o_weights)

        prediction = max(output) #get prediction

        #For Acccuracy 
        o_index = np.where(output==prediction)[0][0] 
        if expected == o_index:
            tp += 1
        else:
            fp += 1
        
        #For confusion matrix. None if training run.
        if c_matrix is not None: #Test Run
            for index in range(10):
                if output[index] == prediction:
                    c_matrix[expected][index] += 1
        else:
            #Training run. Perform weight updates 
            h_weights, o_weights, change_o, change_h = backwards(hidden, output, answer, h_weights, o_weights, row, n, learn, momentum, change_o, change_h)
        i += 1

    epoch_acc = calc_accuracy(tp,fp) #calculate accuracy for the epoch

    #Data output for tracking progress and return needed values
    if c_matrix is None: #this is a training run
        print(f"Train Accuracy: {epoch_acc} | N {n}, Momentum {momentum}")
        return epoch_acc, h_weights, change_h, o_weights, change_o 
    else: #this is a testing run
        print(f"Test Accuracy: {epoch_acc} | N {n}, Momentum {momentum}")
        return epoch_acc, c_matrix

#Function that runs 50 epochs and keeps track of accuracy and confusion matrices for plotting
def run_epoch(n, h_weights, o_weights, change_o, change_h, train_ans, test_ans, train_data, test_data, learn, momentum):
    c_matrix = np.zeros((10, 10)) #fills 10 x 10 matrix filled with 0s for confusion matrix
    train_accuracy_plot = [] #stores plotting points for test accuracy graph
    test_accuracy_plot = [] #stores plotting points for test accuracy graph
    
    train_accuracy_plot.append(calc_zero_accuracy(n, h_weights, o_weights, train_ans, train_data)) #calculate 0th accuracy before starting

    #Run 50 epochs
    for counter in range(50):
        print(f"N {n} Momentum {momentum} Epoch {counter + 1}")

        train_acc, h_weights, change_h, o_weights, change_o  = neural_network(n, h_weights, o_weights, change_o, change_h, train_ans, train_data, learn, momentum) 
        test_acc, c_matrix = neural_network(n, h_weights, o_weights, change_o, change_h, test_ans, test_data, learn, momentum, c_matrix)

        train_accuracy_plot.append(train_acc)
        test_accuracy_plot.append(test_acc)

    return train_accuracy_plot, test_accuracy_plot, c_matrix

def main():
    learn = 0.1 #Default learn value
    momentum = 0.9 #Default momentum value

    #Load and process data
    print("Loading and preprocessing data")
    train_data, train_ans = csv_load(train_path)
    test_data, test_ans = csv_load(test_path)
    print("Done")

    #For Multiprocessing
    max_pool_size = mp.cpu_count()

    #Experiment 1. Varying number of neurons in hidden output. Momentum remains 0.9
    n2 = [20, 50, 100] 
    
    pool = mp.Pool(min(max_pool_size, len(n2)))
    result = pool.starmap(run_epoch_pal, [(n, momentum, learn, train_data, train_ans, test_data, test_ans) for n in n2])
    pool.close()
    
    print(f"N {n2}: {result}")

    #Experiment 2. Varying momentum value. Neuron number in the hidden layer stays at 100
    momentums = [0, 0.25, 0.5] 
    
    pool = mp.Pool(min(max_pool_size, len(momentums)))
    result = pool.starmap(run_epoch_pal, [(n2[2], m, learn, train_data, train_ans, test_data, test_ans) for m in momentums])
    pool.close()

    print(result)

    #Experiment 3. Varying data size. Neuron number in the hidden layer stays at 100 and momentum stays at 0.9
    sizes = [30000, 15000]

    pool = mp.Pool(min(max_pool_size, len(sizes)))
    result = pool.starmap(run_epoch_pal, [(n2[2], momentum, learn, train_data[0:s], train_ans[0:s], test_data, test_ans) for s in sizes])
    pool.close()

    print(result)

if __name__ == '__main__':
    main()