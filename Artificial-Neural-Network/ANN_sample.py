# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:09:05 2019

@author: user
"""
# =============================================================================
# The program is the sample of ANN. There are many methods improving the efficiency of ANN. 
# In initiailizing parameters, Xaiver initialization is applied.
# In forward propagation, ReLU function is applied in input and hidden layers and Sigmoid function is applied in output layer.
# In cost function, binary cross-entropy function is applied.
# In back propagation, Adam optimizer is applied.
# In training ANN, mini-batch gradient descent is applied.
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import time
np.set_printoptions(threshold=np.inf)


NTier = int(4)          
NCol = int(6)
NumCon = int(10)        #The number of container is uncessnary here
np.random.seed(7)       #Deciding a seed for randomizing the sequence of dataset and initializing the parameter in random number


# =============================================================================
# Loading the file of input data, randomizing the sequence of input data, and splitting the input data into 3 sets: Training set, Validation set, and Testing set
# =============================================================================
with open('input_layer_4_6_10_2.pickle', 'rb') as file:
    X = pickle.load(file)
    #print(X)
    X = np.transpose(X)                                                #Transposing the input data for applying the ANN
    permutation = list(np.random.permutation(X.shape[1]))              #Setting a list that contains lots of number (from 0 to number of data -1) and randomizing the sequence
    X = X[:, permutation]                                              #Applying the list permutation to randomizing the input data
    X_valid = X[:, 100000:101000]                                      #Creating the validation set
    print(X_valid.shape)
    X_test = X[:, 101000:102000]                                       #Creating the testing set
    X = X[:, 0:100000]                                                 #Creating the training set
    print(X.shape) 

# =============================================================================
# Loading the file of output data, randomizing the sequence of output data, and splitting the input data into 3 sets: Training set, Validation set, and Testing set  
# =============================================================================
with open('output_layer_4_6_10_2.pickle', 'rb') as file:
    Y = pickle.load(file)
    #permutation = list(np.random.permutation(Y.shape[1]))
    Y = Y[:, permutation]                                              #Applying the list permutation to randomizing the output data
    Y_valid = Y[:, 100000:101000]                                      #Creating the validation set
    print(Y_valid.shape)
    Y_test = Y[:, 101000:102000]                                       #Creating the testing set
    print(Y_test.shape)
    Y = Y[:, 0:100000]                                                 #Creating the training set
    print(Y.shape)
    #print(np.transpose(Y))

#print(np.transpose(X))
#print(np.transpose(Y))
Hidden_layer = int(input("Input Hidden Layer:"))                       #Input the number of hidden layer
T_Layer = []                                                           #list for putting the number of perceptrons in each layer
#np.random.seed(2)
print(X.shape)
n_x = X.shape[0]                                                       #Getting the number of perceptrons of input layer
n_y = Y.shape[0]                                                       #Getting the number of perceptrons of output layer

# =============================================================================
# The if-elif condition puts the number of perceptrons in each layer into T_layer
# =============================================================================
if Hidden_layer > 0:    
    for i in range(0, Hidden_layer):                                   #The for loop records the number of perceptron in each hidden layer
        i = int(input("Input nodes of Hidden Layer :"))
        T_Layer.append(i)
    T_Layer.insert(0, n_x)                                             #Recording the number of perceptron in input layer
    T_Layer.append(n_y)                                                #Recording the number of perceptron in output layer
        
else:
    T_Layer.insert(0, n_x)
    T_Layer.append(n_y)
    
start = time.time()        
print('Total Layer:',T_Layer)

# =============================================================================
# The function initializes the parameters
# =============================================================================
def initialize_parameter(n_x, T_Layer, n_y):                           
    Parameters = {}
    L = len(T_Layer)
    for l in range(1,L):
        Parameters['W' + str(l)] = np.random.randn(T_Layer[l], T_Layer[l-1]) * np.sqrt(1 / T_Layer[l-1])  #Initilizing parameters by Xaivar initialization and 
        Parameters['b' + str(l)] = np.zeros((T_Layer[l], 1))                                              #determining the size of parameters accroding to the number of perceptrons between layers 
    
        assert(Parameters['W' + str(l)].shape == (T_Layer[l], T_Layer[l-1]))   
        assert(Parameters['b' + str(l)].shape == (T_Layer[l], 1))
    
    return Parameters

# =============================================================================
# The function is Sigmoid function
# =============================================================================
def sigmoid(Z):
    A = 1/ (1+np.exp(-Z))
    
    return A

# =============================================================================
# The funtion is Sigmoid function, but it's uncessnary here
# =============================================================================
def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    
    return A
# =============================================================================
# The funtion is linear function, but it's uncessnary here
# =============================================================================
def linear(Z):
    A = Z
    return A

# =============================================================================
# The function is ReLU function
# =============================================================================
def ReLu(Z):
   
    return Z * (Z>0)

# =============================================================================
# The function is softmax function for applying in output layer, but it's uncessnary here
# =============================================================================
def softmax(Z):
    t = np.exp(Z)
    sum_t = np.sum(t, axis = 0)
    A = t / sum_t
    return A
# =============================================================================
# The function is for creating mini-batches and randomize the sequence of dataset to ensure each mini-batch similar distribution
# =============================================================================
def random_mini_batches(X,Y, seed, mini_batch_size = 1024): #(input data, output data, seed for randomizing, the size of mini-batch)
    np.random.seed(seed)
    m = X.shape[1]                                                                    #number of dataset
    mini_batches = []                                                                 #for putting each mini-batch
    permutation = list(np.random.permutation(m))                                      #Setting a list that contains lots of number (from 0 to number of data -1) and randomizing the sequence
    shuffled_X = X[:,permutation]                                                     #Applying the list permutation to randomizing the input data
    shuffled_Y = Y[:,permutation]                                                     #Applying the list permutation to randomizing the output data
    num_complete_mini_batches = math.floor(m/mini_batch_size)                         #Counting number of mini-batches with complete mini-batch size
    # =============================================================================
    # The for loop split training set into many mini-batches
    # =============================================================================
    for i in range(0, num_complete_mini_batches):
        mini_batch_X = shuffled_X[:, mini_batch_size * i : mini_batch_size * (i+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * i : mini_batch_size * (i+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # =============================================================================
    # The if condition form a complete mini-batch which contains the rest of dataset when they are imcomplete 
    # =============================================================================
    if m %  mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_mini_batches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_mini_batches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
        

# =============================================================================
# The function is forward propagation
# =============================================================================
def forward_propogation(X, T_Layer, Parameters, Y): 
    caches = {}                                                              #Storing computational result in each layer for the following backpropagation 
    L = len(T_Layer)                                                         #Number of all layers
    # =============================================================================
    # The if-else condition checks whether there is any hidden layer or not and applies different activation funtion (only calculating the first layer)
    # =============================================================================
    if L == 2:                                                               #If there isn't any hidden layer, the activation function turns to be Sigmoid function for computing from input layer to output layer
        caches['Z'+ str(1)] = np.dot(Parameters['W1'], X) + Parameters['b1'] #Z1 = W * X + b
        #caches['A'+str(1)] = ReLu(caches['Z'+ str(1)])
        #caches['A'+str(1)] = softmax(caches['Z'+ str(1)])
        caches['A'+str(1)] = sigmoid(caches['Z'+ str(1)])                    #A1 = Sigmoid(Z1)
        A = caches['A'+str(1)]
        caches['A'+str(0)] = X                                               #A0 = X
        
    else:                                                                    #If the is any hidden layer, the program doesn't use Sigmiod function in the activation function
        caches['Z'+ str(1)] = np.dot(Parameters['W1'], X) + Parameters['b1'] #Z1 = W1 * X + b1
        #caches['A'+str(1)] = sigmoid(caches['Z'+ str(1)])
        caches['A'+str(1)] = ReLu(caches['Z'+ str(1)])                       #A1 = ReLU(Z1)
        A = caches['A'+str(1)]
        caches['A'+str(0)] = X                                               #A0 = X
    # =============================================================================
    # The condition keep calculating values from the second layer to the last layer
    # =============================================================================
    if L > 2 :                                                               #Ensuring the if condition to work with hidden layer 
        # =============================================================================
        # The for loop calculates the value of Z and A until the last layer        
        # =============================================================================
        for l in range(2, L - 1):                                            
            A_prev = A
            caches['Z' + str(l)] = np.dot(Parameters['W'+str(l)], A_prev) + Parameters['b'+str(l)]   #Z(l) = W (l) * A (l-1) + b (l)
            
            #caches['A' + str(l)] = sigmoid(caches['Z' + str(l)])
            caches['A'+str(l)] = ReLu(caches['Z'+ str(l)])                                           #A(l) = ReLU(Z(l))

            A = caches['A'+str(l)]
        
            
        A_prev = A
        caches['Z' + str(L-1)] = np.dot(Parameters['W'+str(L-1)], A_prev) + Parameters['b'+str(L-1)] #Z(L) = W(L) * A(L-1) + b(L)
        caches['A' + str(L-1)] = sigmoid(caches['Z' + str(L-1)])                                     #A(l) = Sigmoid(Z(l))
        #caches['A' + str(L-1)] = softmax(caches['Z' + str(L-1)])
        A = caches['A'+str(L-1)]
    

  
    assert (A.shape == (Y.shape[0], X.shape[1]))  #(node of Y,m)

    return A, caches
'''
def batch_norm(Z):
    mu = np.mean(Z, axis = 1)
'''   
# =============================================================================
# The function is cost function (binary cross-entropy)
# =============================================================================
def cost_function(A,Y, lambd = 0.7):
    m = Y.shape[1]
    n = Y.shape[0]
    
    Logi_reg = np.multiply(Y, np.log(A + 1e-8)) + np.multiply((1-Y),np.log(1-A + 1e-8))  
    cost = -(np.sum(Logi_reg))/ m

    #Logi_reg = -1 * np.sum(np.multiply(Y, np.log(A))) / m 
    #cost = Logi_reg
    #print('cost =',cost)
   
    #Line_reg = ((A-Y)**2)* 1/2 
    '''  
    sum_w_squ = 0
    for i in range(1, int(len(Parameters) / 2 + 1)):
        sum_w_squ += np.sum(Parameters['W' + str(i)] ** 2)
    L2_regularization_cost = (lambd / (m * 2)) * sum_w_squ
    '''
    
    #cost = Logi_reg + L2_regularization_cost
    #cost = np.sum(Line_reg)/ (m*n) + L2_regularization_cost


    '''
    cost = -(np.sum(Logi_Reg,axis = 1,keepdims = True))/m    #for node of Y != 1
    '''
    cost = np.squeeze(cost)
    #print(cost)
    return cost


# =============================================================================
# The function is back propagation (partial derivative)
# =============================================================================
def Back_propogation(Parameters, caches, X, Y, A, T_Layer, lambd = 0.7):                    
    grads = {}
    L = len(T_Layer)
    m = Y.shape[1]                               #Number of data
    
    dA = -Y/(A + 1e-8) + (1-Y)/(1-A + 1e-8)      #Doing partial derivative on A(l)
    #dA = (A - Y) / (A * (1-A))
    
    #dA = - (Y / A)                              #multi-class cross-entropy
    #dA = A - Y                                  #linear_regression 
    dZ = dA * A *(1-A)                           #Doing partial derivative on Z(l) = dA * derivative of sigmoid function 
    #dZ = A - Y                                  #softmax and cross-entropy
    #dZ = dA * 1 * (A > 0)                       #ReLU

    grads['dW'+str(L-1)]= np.dot(dZ, np.transpose(caches['A'+str(L-2)]))/ m + ((lambd / m) * Parameters['W'+str(L-1)]) #regularization, but the progarm does't apply it. 
                                                                                                                       #Thus, there are only do partial devirative on W, dW = np.dot(dZ,A.T) 
    #grads['dW'+str(L-1)]= np.dot(dZ, np.transpose(caches['A'+str(L-2)]))/ m
    grads['db'+str(L-1)] = np.sum(dZ,axis = 1, keepdims =True)/m                                                       #Partial devirative on b
  
    if L > 2 :
        
        for l in range(1, L-1):
            dZ_prev = dZ
            dA = np.dot(np.transpose(Parameters['W'+str(-l+L)]),dZ_prev)
            dA_prev = dA
            #dZ = dA_prev * caches['A'+str(-l+L-1)] * (1-caches['A'+str(-l+L-1)])                                                         #sigmoid
            
            dZ = dA_prev * 1 * (caches['A'+str(-l+L-1)] > 0)                                                                              #ReLU
            grads['dW'+str(-l+L-1)] = np.dot(dZ, np.transpose(caches['A'+ str(-l+L-2)])) /m + ((lambd / m) * Parameters['W'+str(-l+L-1)]) #dW = np.dot(dZ,A.T) 
            #grads['dW'+str(-l+L-1)] = np.dot(dZ, np.transpose(caches['A'+ str(-l+L-2)])) /m
            grads['db'+str(-l+L-1)] = np.sum(dZ,axis = 1, keepdims =True) / m                                                             #sum the dataset, so axis = 1

    
    return grads

# =============================================================================
# The function initializes hyperparameters for the Adam optimizer
# =============================================================================
def hyperparameter(T_Layer, Parameters):
    v = {}
    s = {}
    # =============================================================================
    # Copying the same size of parameters to create hyperparameters and initialize them as 0 
    # =============================================================================
    for i in range(1, len(T_Layer)):
        v['dW'+str(i)] = np.zeros((Parameters['W'+str(i)].shape))
        v['db'+str(i)] = np.zeros((Parameters['b'+str(i)].shape))
        s['dW'+str(i)] = np.zeros((Parameters['W'+str(i)].shape))
        s['db'+str(i)] = np.zeros((Parameters['b'+str(i)].shape))
    return v, s 

# =============================================================================
# The function shows three different optimizers : Momentum, RMSprop, and Adam. The program follows Adam optimizer to update parameters 
# =============================================================================
def update_parameters(optim_algorithm, Parameters, grads, T_Layer, v, s, t, beta_1 = 0.9 , beta_2 = 0.999, epsilson = 1e-8, learning_rate = 10):
    L = len(T_Layer)
    v_correction = {}
    s_correction = {}
    
    if optim_algorithm == "Momentum":
        
        for i in range(1,L):
            v['dW'+str(i)] = beta_1 * v['dW'+str(i)] + (1-beta_1) * grads['dW'+str(i)]
            v['db'+str(i)] = beta_1 * v['db'+str(i)] + (1-beta_1) * grads['db'+str(i)]
            v_correction['dW'+str(i)] = v['dW'+str(i)] / (1-(beta_1**t))
            v_correction['db'+str(i)] = v['db'+str(i)] / (1-(beta_1**t))
            Parameters['W'+str(i)] = Parameters['W'+str(i)] - v_correction['dW'+str(i)] * learning_rate 
            Parameters['b'+str(i)] = Parameters['b'+str(i)] - v_correction['db'+str(i)] * learning_rate
            
    if optim_algorithm == "RMSprop":
        
        for i in range(1,L): 
            s['dW'+str(i)] = beta_2 * s['dW'+str(i)] + (1-beta_2) * (grads['dW'+str(i)]**2)
            s['db'+str(i)] = beta_2 * s['db'+str(i)] + (1-beta_2) * (grads['db'+str(i)]**2)
            s_correction['dW'+str(i)] = s['dW'+str(i)] / (1-(beta_2**t))
            s_correction['db'+str(i)] = s['db'+str(i)] / (1-(beta_2**t))
            Parameters['W'+str(i)] = Parameters['W'+str(i)] - grads['dW'+str(i)] / (np.sqrt(s_correction['dW'+str(i)]) + epsilson) * learning_rate 
            Parameters['b'+str(i)] = Parameters['b'+str(i)] - grads['db'+str(i)] / (np.sqrt(s_correction['db'+str(i)]) + epsilson) * learning_rate
      
    if optim_algorithm == "Adam":
        
        for i in range(1,L): 
            v['dW'+str(i)] = beta_1 * v['dW'+str(i)] + (1-beta_1) * grads['dW'+str(i)]
            v['db'+str(i)] = beta_1 * v['db'+str(i)] + (1-beta_1) * grads['db'+str(i)]
            s['dW'+str(i)] = beta_2 * s['dW'+str(i)] + (1-beta_2) * (grads['dW'+str(i)]**2)
            s['db'+str(i)] = beta_2 * s['db'+str(i)] + (1-beta_2) * (grads['db'+str(i)]**2)
            v_correction['dW'+str(i)] = v['dW'+str(i)] / (1-(beta_1**t))                    #Bia correction 
            v_correction['db'+str(i)] = v['db'+str(i)] / (1-(beta_1**t))                    #Bia correction 
            s_correction['dW'+str(i)] = s['dW'+str(i)] / (1-(beta_2**t))                    #Bia correction 
            s_correction['db'+str(i)] = s['db'+str(i)] / (1-(beta_2**t))                    #Bia correction 
            Parameters['W'+str(i)] = Parameters['W'+str(i)] - v_correction['dW'+str(i)]/(np.sqrt(s_correction['dW'+str(i)]) + epsilson) * learning_rate 
            Parameters['b'+str(i)] = Parameters['b'+str(i)] - v_correction['db'+str(i)]/(np.sqrt(s_correction['db'+str(i)]) + epsilson) * learning_rate
    

    return Parameters

# =============================================================================
# The function helps to set thresold to predict the result of output data 
# =============================================================================
def predict(Parameters, T_Layer ,X):                  
    A,caches = forward_propogation(X, T_Layer, Parameters, Y)                   #Applying input data in forward propagation to get predicted A            
    #A = np.where(A>=0.5,1,0)
    #print(A)

    # =============================================================================
    # The for loop make prediction which is turning the value in A into binary integers    
    # =============================================================================
    for j in range(0, A.shape[1]):  #Each predicted A. size = (12, 1)                                      
        Max_take = 0
        Max_put = 0
        # =============================================================================
        # The for loop finds the largest number among the first half of predicted A         
        # =============================================================================
        for take in range(0, 6):    
            if A[take][j] >= Max_take:
                Max_take = A[take][j]
                Max_take_row = take
                
        # =============================================================================
        # The for loop finds the largest number among the second half of predicted A         
        # =============================================================================
        for put in range(6, 12):

            if A[put][j] >= Max_put:
                Max_put = A[put][j]
                Max_put_row = put
                
        
        for i in range(0, A.shape[0]):    #All number in predicted A are turned into 0 
            A[i][j] = 0 
            
        A[Max_take_row][j] = 1            #The position of largest number is turned into 1
        A[Max_put_row][j] = 1             #The position of largest number is turned into 1
        
    #A = np.where(A>=0.5,1,0)
    #print(A)
    predictions = A
    return predictions

# =============================================================================
# The function updates parameters by repeatly executing forward propagation and back propagation 
# =============================================================================
def nn_model(X,Y,num_epochs = 500000, optim_algorithm = "Adam", beta_1 = 0.9, beta_2 = 0.999, epsilson = 1e-8, learning_rate = 0.0035, print_cost = False):
    seed = 10
    Parameters = initialize_parameter(n_x, T_Layer, n_y)              #Randomly initializing the parameters, the value is between 0 to 1 
    v,s = hyperparameter(T_Layer, Parameters)                         #Initializing hyperparameters to 0
    costs = []                                                        #For recording the change of cost

    #seed = 10
    t = 0                                                             #Applying in Adam optimizer
    # =============================================================================
    # The for loop repeatly runs the forward propagation and backpropagation      
    # =============================================================================
    for i in range(0, num_epochs + 1):
        seed = seed + 1                                               #Changing the seed to ensure the different sequence on  mini-batches 
        minibatches = random_mini_batches(X,Y, seed ,mini_batch_size = 1024) #Creating mini-batches
        # =============================================================================
        # The for loop uses every mini-batch to update parameters         
        # =============================================================================
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch                                                         #Loading input data and output data
            A, caches = forward_propogation(minibatch_X, T_Layer, Parameters, minibatch_Y)                 #Executing forward propagation
        
            cost = cost_function(A,minibatch_Y, lambd = 0.0)                                               #Executing cost functioin
 
            grads = Back_propogation(Parameters, caches, minibatch_X, minibatch_Y, A, T_Layer, lambd = 0.0)#Executing back propagation
            t = t + 1
            Parameters = update_parameters(optim_algorithm, Parameters, grads, T_Layer, v, s, t, beta_1, beta_2, epsilson, learning_rate) #Updating parameters

        # =============================================================================
        # The if condition calculates the accuracy of training set in each 10 epochs
        # =============================================================================
        if print_cost and i % 10 == 0:
            predictions = predict(Parameters, T_Layer, X)                       #Getting the prediceted A
            error = predictions - Y                                             #Calculating the error
            error = abs(error)                                                  #Adding absolute value on error 
            error = np.transpose(error)                                         #(12, m) to (m, 12) m = number of data in training set
            error = np.sum(error, axis = 1)                                     #Summing the value in axis 1
            error = np.where(error == 0, 1, 0)                                  #Check each column. If error = 0, the value of column will be 1 which means 
                                                                                #the prediction in this data is correct, else: 0
            accuracy = np.sum(error)/ (error.shape[0])                          #Calculating the accuracy
            print("Cost after epoch %i: %f" %(i, cost), 'Accuracy =', '%.5f' %accuracy)    #%i, %f
        if print_cost and i % 10 == 0:
            costs.append(cost)

    #plt the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('num_epochs (per 10)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return Parameters

Parameters = nn_model(X,Y,num_epochs = 100,print_cost = True)   #The updated parameters after 100 epochs

L = len(T_Layer)


with open('weight_Min_max_4_6_10_2.pickle','wb') as file:    
    pickle.dump(Parameters,file)


predictions_valid = predict(Parameters, T_Layer, X_valid)       #Calculating the predict A of validation set
predictions_valid_T = np.transpose(predictions_valid)


error_v = predictions_valid - Y_valid                           #Calculating the error 
error_v = abs(error_v)                                          #Adding absolute value on error
error_v = np.transpose(error_v)                                 #(12, m) to (m, 12) m = number of data in validation set
#print('error_v = ', error_v)
error_v = np.sum(error_v, axis = 1)                             #Summing the value in axis 1
error_v = np.where(error_v == 0, 1, 0)                          #Check each column. If error = 0, the value of column will be 1 which means 
                                                                #the prediction in this data is correct, else: 0
print('Valid Accurate rate = ', np.sum(error_v)/ (error_v.shape[0]) * 100 ,'%')


predictions_test = predict(Parameters, T_Layer, X_test)         #Calculating the predict A of testing set
predictions_test_T = np.transpose(predictions_test)


error_t = predictions_test - Y_test                             #Calculating the error 
error_t = abs(error_t)                                          #Adding absolute value on error
error_t = np.transpose(error_t)                                 #(12, m) to (m, 12) m = number of data in testing set
error_t = np.sum(error_t, axis = 1)                             #Summing the value in axis 1
error_t = np.where(error_t == 0, 1, 0)                          #Check each column. If error = 0, the value of column will be 1 which means 
                                                                #the prediction in this data is correct, else: 0
print('Test Accurate rate = ', np.sum(error_t)/ (error_t.shape[0]) * 100 ,'%')

predictions = predict(Parameters,T_Layer, X)
predictions_T = np.transpose(predictions)
Y_T = np.transpose(Y)

error = predictions - Y                                          #abs =絕對值 Calculating the error of training set

error_T = np.transpose(error)


#print(error_T) 
wrong_taking = 0                                                 #Calculating the number of decision on choosing wrong column to take container
didnt_take = 0                                                   #Calculating the number of decision on column it should choose to take container but didn't do it  
wrong_puting = 0                                                 #Calculating the number of decision on choosing wrong column to put container
didnt_put = 0                                                    #Calculating the number of decision on column it should choose to put container but didn't do it

# =============================================================================
# The following two for loop count the number of miatake that the ANN didn't predict well
# =============================================================================
for i in range(0, error_T.shape[0]):
    for j in range(0, NCol):
        if error_T[i][j] == 1:
            wrong_taking += 1
        if error_T[i][j] == -1 :
            didnt_take += 1
    
for i in range(0, error_T.shape[0]):
    for j in range(NCol, NCol*2):
        if error_T[i][j] == 1:
            wrong_puting += 1
        if error_T[i][j] == -1 :
            didnt_put += 1
            
#The following 4 lines are for calculating the accuracy of training set. The method is the same to the previous lines
error = abs(error)                                           
error = np.transpose(error)
error = np.sum(error, axis = 1)
#print(error)
error = np.where(error == 0, 1, 0)
#print(error)
print(error.shape)

print('Accurate rate = ', np.sum(error)/ (error.shape[0]) * 100 ,'%')
print('wrong_taking = ', wrong_taking)
print('didnt_take = ', didnt_take)
print('wrong_puting = ', wrong_puting)
print('didnt_put = ', didnt_put)


stop = time.time()
print('Computational time =',(stop - start))