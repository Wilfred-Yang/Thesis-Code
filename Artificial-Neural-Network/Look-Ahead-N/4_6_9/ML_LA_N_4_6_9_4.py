# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:29:54 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import time
np.set_printoptions(threshold=np.inf)


NTier = int(4)
NCol = int(6)
NumCon = int(9)
np.random.seed(7)

with open('input_layer_4_6_9_4.pickle', 'rb') as file:
    X = pickle.load(file)
    #print(X)
    
    X = np.transpose(X)
    permutation = list(np.random.permutation(X.shape[1]))
    #print(permutation)
    X = X[:, permutation]
    X_valid = X[:, 100000:101000]
    print(X_valid.shape)
    X_test = X[:, 101000:102000]
    X = X[:, 0:100000]
    print(X.shape)
with open('output_layer_4_6_9_4.pickle', 'rb') as file:
    Y = pickle.load(file)
    #permutation = list(np.random.permutation(Y.shape[1]))
    #print(permutation)
    Y = Y[:, permutation]
    Y_valid = Y[:, 100000:101000]
    print(Y_valid.shape)
    Y_test = Y[:, 101000:102000]
    print(Y_test.shape)
    Y = Y[:, 0:100000]
    print(Y.shape)
    #print(np.transpose(Y))

#print(np.transpose(X))
#print(np.transpose(Y))
Hidden_layer = int(input("Input Hidden Layer:"))
T_Layer = []  #list for Hidden Layer
#np.random.seed(2)
print(X.shape)
n_x = X.shape[0]
n_y = Y.shape[0]
if Hidden_layer > 0:    
    for i in range(0, Hidden_layer):
        i = int(input("Input nodes of Hidden Layer :"))
        T_Layer.append(i)
    T_Layer.insert(0, n_x)
    T_Layer.append(n_y)
        
else:
    T_Layer.insert(0, n_x)
    T_Layer.append(n_y)
    
start = time.time()        
print('Total Layer:',T_Layer)

def initialize_parameter(n_x, T_Layer, n_y):
    Parameters = {}
    L = len(T_Layer)
    for l in range(1,L):
        Parameters['W' + str(l)] = np.random.randn(T_Layer[l], T_Layer[l-1]) * np.sqrt(1 / T_Layer[l-1]) 
        Parameters['b' + str(l)] = np.zeros((T_Layer[l], 1))

        assert(Parameters['W' + str(l)].shape == (T_Layer[l], T_Layer[l-1]))   
        assert(Parameters['b' + str(l)].shape == (T_Layer[l], 1))
    
    return Parameters

def sigmoid(Z):
    A = 1/ (1+np.exp(-Z))
    
    return A

def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    
    return A

def linear(Z):
    A = Z
    return A

def ReLu(Z):
   
    return Z * (Z>0)

def softmax(Z):
    t = np.exp(Z)
    sum_t = np.sum(t, axis = 0)
    A = t / sum_t
    return A

def random_mini_batches(X,Y, seed, mini_batch_size = 1024):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    num_complete_mini_batches = math.floor(m/mini_batch_size) 
    
    for i in range(0, num_complete_mini_batches):
        mini_batch_X = shuffled_X[:, mini_batch_size * i : mini_batch_size * (i+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * i : mini_batch_size * (i+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m %  mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_mini_batches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_mini_batches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
        

def forward_propogation(X, T_Layer, Parameters, Y):
    caches = {}
    L = len(T_Layer)

    if L == 2:
        caches['Z'+ str(1)] = np.dot(Parameters['W1'], X) + Parameters['b1']
        #caches['A'+str(1)] = ReLu(caches['Z'+ str(1)])
        #caches['A'+str(1)] = softmax(caches['Z'+ str(1)])
        caches['A'+str(1)] = sigmoid(caches['Z'+ str(1)])
        A = caches['A'+str(1)]
        caches['A'+str(0)] = X
        
    else:
        caches['Z'+ str(1)] = np.dot(Parameters['W1'], X) + Parameters['b1']
        #caches['A'+str(1)] = sigmoid(caches['Z'+ str(1)])
        caches['A'+str(1)] = ReLu(caches['Z'+ str(1)])
        A = caches['A'+str(1)]
        caches['A'+str(0)] = X
    
    if L > 2 :
        
        for l in range(2, L - 1):
            A_prev = A
            caches['Z' + str(l)] = np.dot(Parameters['W'+str(l)], A_prev) + Parameters['b'+str(l)]
            
            #caches['A' + str(l)] = sigmoid(caches['Z' + str(l)])
            caches['A'+str(l)] = ReLu(caches['Z'+ str(l)])

            A = caches['A'+str(l)]
        
            
        A_prev = A
        caches['Z' + str(L-1)] = np.dot(Parameters['W'+str(L-1)], A_prev) + Parameters['b'+str(L-1)]
        caches['A' + str(L-1)] = sigmoid(caches['Z' + str(L-1)])
        #caches['A' + str(L-1)] = softmax(caches['Z' + str(L-1)])
        A = caches['A'+str(L-1)]
    

  
    assert (A.shape == (Y.shape[0], X.shape[1]))  #(node of Y,m)

    return A, caches
'''
def batch_norm(Z):
    mu = np.mean(Z, axis = 1)
'''   
def cost_function(A,Y, lambd = 0.7):
    m = Y.shape[1]
    n = Y.shape[0]
    
    Logi_reg = np.multiply(Y, np.log(A + 1e-8)) + np.multiply((1-Y),np.log(1-A + 1e-8))  #we can directly multiply it 
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


    
def Back_propogation(Parameters, caches, X, Y, A, T_Layer, lambd = 0.7):
    grads = {}
    L = len(T_Layer)
    m = Y.shape[1]
    
    dA = -Y/(A + 1e-8) + (1-Y)/(1-A + 1e-8)  #logistic regression for only one neuron
    #dA = (A - Y) / (A * (1-A))
    
    #dA = - (Y / A)         # mutli-class cross-entropy
    #dA = A - Y              #linear_regression 
    dZ = dA * A *(1-A)     #sigmoid
    #dZ = A - Y             #softmax and cross-entropy
    #dZ = dA * 1 * (A > 0)   #ReLU

    grads['dW'+str(L-1)]= np.dot(dZ, np.transpose(caches['A'+str(L-2)]))/ m + ((lambd / m) * Parameters['W'+str(L-1)]) #regularization
    #grads['dW'+str(L-1)]= np.dot(dZ, np.transpose(caches['A'+str(L-2)]))/ m
    #grads['dW'+str(L-1)]= np.dot(dZ, np.transpose(caches['A'+str(L-2)]))/ m
    grads['db'+str(L-1)] = np.sum(dZ,axis = 1, keepdims =True)/m
  
    if L > 2 :
        
        for l in range(1, L-1):
            dZ_prev = dZ
            dA = np.dot(np.transpose(Parameters['W'+str(-l+L)]),dZ_prev)
            dA_prev = dA
            #dZ = dA_prev * caches['A'+str(-l+L-1)] * (1-caches['A'+str(-l+L-1)])             #sigmoid
            
            dZ = dA_prev * 1 * (caches['A'+str(-l+L-1)] > 0)                                 #ReLU
            grads['dW'+str(-l+L-1)] = np.dot(dZ, np.transpose(caches['A'+ str(-l+L-2)])) /m + ((lambd / m) * Parameters['W'+str(-l+L-1)]) #regularization
            #grads['dW'+str(-l+L-1)] = np.dot(dZ, np.transpose(caches['A'+ str(-l+L-2)])) /m
            #grads['dW'+str(-l+L-1)] = np.dot(dZ, np.transpose(caches['A'+ str(-l+L-2)])) /m   #dW = np.dot(dZ,A.T) 
            grads['db'+str(-l+L-1)] = np.sum(dZ,axis = 1, keepdims =True) / m                 #sum the dataset, so axis = 1

    
    return grads


def hyperparameter(T_Layer, Parameters):
    v = {}
    s = {}
    for i in range(1, len(T_Layer)):
        v['dW'+str(i)] = np.zeros((Parameters['W'+str(i)].shape))
        v['db'+str(i)] = np.zeros((Parameters['b'+str(i)].shape))
        s['dW'+str(i)] = np.zeros((Parameters['W'+str(i)].shape))
        s['db'+str(i)] = np.zeros((Parameters['b'+str(i)].shape))
    return v, s 

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
            v_correction['dW'+str(i)] = v['dW'+str(i)] / (1-(beta_1**t))
            v_correction['db'+str(i)] = v['db'+str(i)] / (1-(beta_1**t))
            s_correction['dW'+str(i)] = s['dW'+str(i)] / (1-(beta_2**t))
            s_correction['db'+str(i)] = s['db'+str(i)] / (1-(beta_2**t))
            Parameters['W'+str(i)] = Parameters['W'+str(i)] - v_correction['dW'+str(i)]/(np.sqrt(s_correction['dW'+str(i)]) + epsilson) * learning_rate 
            Parameters['b'+str(i)] = Parameters['b'+str(i)] - v_correction['db'+str(i)]/(np.sqrt(s_correction['db'+str(i)]) + epsilson) * learning_rate
    

    return Parameters


def predict(Parameters, T_Layer ,X):    
    A,caches = forward_propogation(X, T_Layer, Parameters, Y)
    #A = np.where(A>=0.5,1,0)
    #print(A)

    
    for j in range(0, A.shape[1]):
        Max_take = 0
        Max_put = 0
        for take in range(0, 6):    
            if A[take][j] >= Max_take:
                Max_take = A[take][j]
                Max_take_row = take
                

        for put in range(6, 12):

            if A[put][j] >= Max_put:
                Max_put = A[put][j]
                Max_put_row = put
                
        
        for i in range(0, A.shape[0]):
            A[i][j] = 0 
            
        A[Max_take_row][j] = 1
        A[Max_put_row][j] = 1
     
    #A = np.where(A>=0.5,1,0)
    #print(A)
    predictions = A
    return predictions


def nn_model(X,Y,num_epochs = 500000, optim_algorithm = "Adam", beta_1 = 0.9, beta_2 = 0.999, epsilson = 1e-8, learning_rate = 0.005, print_cost = False):
    
    Parameters = initialize_parameter(n_x, T_Layer, n_y)
    v,s = hyperparameter(T_Layer, Parameters)
    costs = [] #keep the track on the cost
    '''
    high_accuracy = 0
    high_accuracy_epoch = 0
    low_cost = 1
    low_cost_epoch = 0
    '''
    seed = 10
    t = 0
    for i in range(0, num_epochs + 1):
        seed = seed + 1 
        minibatches = random_mini_batches(X,Y, seed ,mini_batch_size = 1024)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            A, caches = forward_propogation(minibatch_X, T_Layer, Parameters, minibatch_Y)
        
            cost = cost_function(A,minibatch_Y, lambd = 0.0)     
 
            grads = Back_propogation(Parameters, caches, minibatch_X, minibatch_Y, A, T_Layer, lambd = 0.0)
            t = t + 1
            Parameters = update_parameters(optim_algorithm, Parameters, grads, T_Layer, v, s, t, beta_1, beta_2, epsilson, learning_rate)


        if print_cost and i % 10 == 0:
            predictions = predict(Parameters, T_Layer, X)
            error = predictions - Y
            error = abs(error)
            error = np.transpose(error)
            error = np.sum(error, axis = 1)
            error = np.where(error == 0, 1, 0)
            accuracy = np.sum(error)/ (error.shape[0])
            print("Cost after epoch %i: %f" %(i, cost), 'Accuracy =', '%.5f' %accuracy)    #%i, %f
        if print_cost and i % 10 == 0:
            costs.append(cost)
    '''
        if cost < low_cost:
            low_cost = cost
            low_cost_epoch = i
        if cost < 0.0005:
            predictions = predict(Parameters, T_Layer, X)
            error = predictions - Y
            error = abs(error)
            error = np.transpose(error)
            error = np.sum(error, axis = 1)
            error = np.where(error == 0, 1, 0)
            accuracy = np.sum(error)/ (error.shape[0])
            if accuracy > high_accuracy:
                high_accuracy = accuracy
                high_accuracy_epoch = i
            
    print('Highest accuracy =', high_accuracy * 100, '%')
    print('epoch for highest accuracy =', high_accuracy_epoch)
    print('low cost =', low_cost)
    print('epoch for low cost =', low_cost_epoch)
    '''
    #plt the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('num_epochs (per 10)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return Parameters

Parameters = nn_model(X,Y,num_epochs = 100,print_cost = True)

L = len(T_Layer)
#for i in range(1,L):
    #print('W',i, " = ", Parameters['W'+str(i)])
    #print('b',i, " = ", Parameters['b'+str(i)])

with open('weight_LA_N_4_6_9_4.pickle','wb') as file:    
    pickle.dump(Parameters,file)


predictions_valid = predict(Parameters, T_Layer, X_valid)
predictions_valid_T = np.transpose(predictions_valid)


error_v = predictions_valid - Y_valid

error_v = abs(error_v)
error_v = np.transpose(error_v)
#print('error_v = ', error_v)
error_v = np.sum(error_v, axis = 1)

error_v = np.where(error_v == 0, 1, 0)
print('Valid Accurate rate = ', np.sum(error_v)/ (error_v.shape[0]) * 100 ,'%')


predictions_test = predict(Parameters, T_Layer, X_test)
predictions_test_T = np.transpose(predictions_test)


error_t = predictions_test - Y_test
error_t = abs(error_t)
error_t = np.transpose(error_t)

error_t = np.sum(error_t, axis = 1)
error_t = np.where(error_t == 0, 1, 0)
print('Test Accurate rate = ', np.sum(error_t)/ (error_t.shape[0]) * 100 ,'%')


predictions = predict(Parameters,T_Layer, X)
predictions_T = np.transpose(predictions)
Y_T = np.transpose(Y)
#print(predictions_T)
#print(Y_T)
'''
print(predictions)
print(Y)
'''

error = predictions - Y  #abs =絕對值
'''
error = abs(error)
error = np.transpose(error)
error = np.sum(error, axis = 1)
error = np.where(error == 0, 1, 0)
print(error.shape)
print('Accurate rate = ', np.sum(error)/ (error.shape[0]) * 100 ,'%')
'''
error_T = np.transpose(error)


#print(error_T) 
wrong_taking = 0
didnt_take = 0
wrong_puting = 0 
didnt_put = 0

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