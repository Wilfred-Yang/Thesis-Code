# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 12:47:55 2018

@author: user
"""

import numpy as np
import pickle
import time
np.set_printoptions(threshold=np.inf)
start = time.time()

NTier = int(4)
NCol = int(6)
NumCon = int(18)

with open ('input_layer_4_6_18.pickle', 'rb') as file:
    input_layer_4_6_18 = pickle.load(file)
    

input_layer = np.transpose(input_layer_4_6_18)

#input_layer = input_layer[:, 613:614]
print(input_layer.shape)
#print(input_layer)
#print(len(np.transpose(input_layer)))
#print(np.transpose(input_layer_4_3_7_4))
input_layer_trans = np.transpose(input_layer)
#print(input_layer_trans)
dataset = np.transpose(input_layer)
dataset.shape = (len(dataset), NTier, NCol)
#print('dataset =', dataset)




def sigmoid(Z):
    A = 1/ (1+np.exp(-Z))
    
    return A

def ReLu(Z):
   
    return Z * (Z>0)

def softmax(Z):
    t = np.exp(Z)
    sum_t = np.sum(t, axis = 0)
    A = t / sum_t
    return A

def forward_propogation(X, L, Parameters):
    caches = {}
    if L == 2:
        caches['Z'+ str(1)] = np.dot(Parameters['W1'], X) + Parameters['b1']
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
        #caches['A' + str(L-1)] = softmax(caches['Z' + str(L-1)])
        caches['A' + str(L-1)] = sigmoid(caches['Z' + str(L-1)])
        A = caches['A'+str(L-1)]
  

    return A, caches

def find_row_top_con(dataset, i, position_1, put_column):
    
    put_row = 4
    for row in range(0, 4):
        if dataset[i][row][int(position_1[1])] != 0:
            take_row = row
            break 
        
    for row in range(0, 4):                     
        if dataset[i][row][put_column] != 0:
            put_row = row

            break

    return take_row, put_row
 
def start_from_one(Bay, NTier, NCol): #let the bay order start from container one
    for row in range(0, NTier):
        for column in range(0, NCol):
            if Bay[row][column] > 0:
                Bay[row][column] = Bay[row][column] - 1
    
    return Bay

def count_last_two_con(Bay):
    position_2 = np.where(Bay == 2)
    position_1 = np.where(Bay == 1)
    if int(position_2[1]) == int(position_1[1]):
        if int(position_1[0]) > int(position_2[0]):
            movement_1_2 = 3
        else:
            movement_1_2 = 2
    else:
        movement_1_2 = 2
    
    return movement_1_2

def Height(Bay):
    height = np.zeros((6,), dtype = int)
    for row in range(0, 4):
        for column in range(0, 6):
            if Bay[row][column] >= 1:
                height[column] += 1
    
    return height

Round = 0
error = 0
dataset_movement = []
error_MM_container = []
error_MM_deadlock = []

for i in range(0, len(dataset)):
    last_take = 0
    movement = 0
    #print(np.sum(dataset[i]))
    #print('dataset =' + str(i), '\n',dataset[i])
    while np.sum(dataset[i]) > 3: #Because there will be only con1 and con2 
    
        high_con = np.max(dataset[i])
        position_1 = np.where(dataset[i] == 1)
        height = Height(dataset[i])
        for row in range(0, 4):
            if dataset[i][row][int(position_1[1])] >= 1:
                row_low = row
                break
        #print('row_low =', row_low)
        #print(high_con, position_1[0] - row_low + 1)
        if int(position_1[0]) - row_low + 1 == 1:
            if row_low == int(position_1[0]):
                dataset[i][int(position_1[0])][int(position_1[1])] = 0 
                dataset[i] = start_from_one(dataset[i], NTier, NCol) 
                last_take = 1
        else:    
            with open("weight_min_max_4_6_" + str(high_con) + '_' + str(int(position_1[0] - row_low + 1)) + '.pickle', 'rb') as file:
                Parameters = pickle.load(file) 
        
        
            L = int(len(Parameters) / 2 + 1)
            x = input_layer[:, i:i+1]
            #print(x)
            A, cache = forward_propogation(x, L, Parameters)
            #print(A)
            #A = np.where(A>=0.5,1,0)
            
            for j in range(0, A.shape[1]):
                Max_take = 0
                for take in range(0, 6):    
                    if A[take][j] >= Max_take:
                        Max_take = A[take][j]
                        Max_take_row = take
                
                for k in [x for x in range(0, 6) if x != Max_take_row]:
                    A[k][j] = 0        
                A[Max_take_row][j] = 1  
            
            for j in range(0, A.shape[1]):
                Max_put = 0
                for put in range(6, 12):
                    if A[put][j] >= Max_put:
                        Max_put = A[put][j]
                        Max_put_row = put
                for k in [x for x in range(6, 12) if x != Max_put_row]:
                    A[k][j] = 0        
                A[Max_put_row][j] = 1
            
            #print(A)
            #print(len(A))

            for column in range(0, 6):
                if A[column] == 1:
                    take_column = column  
                    break
            #print('take_column =', take_column)
            
            for column in range(6, 12):
                if A[column] == 1:
                    put_column = column
                    break
            put_column = put_column - 6 
            
            for row in range(0, 4):                            
                if dataset[i][row][put_column] >= 1 :
                    put_row = row
                    break
                #print('put_row =', put_row)
            if dataset[i][NTier -1][put_column] == 0:
                put_row = NTier
                            
            for row in range(0, 4):
                if dataset[i][row][take_column] >= 1:
                    take_row = row
                    break
            #print('Last_take =', last_take)
            #print('Take_row =', take_row)
            #print('Take_column =', take_column)
            #print('Put_row =', put_row)
            #print('Put_column =', put_column)
            if dataset[i][0][put_column] == 0 and dataset[i][int(position_1[0])][put_column] != 1 and dataset[i][3][take_column] != 0 and dataset[i][take_row][take_column] != last_take and take_column != put_column:                       
                                        
                dataset[i][put_row - 1][put_column] = dataset[i][take_row][take_column]
                last_take = dataset[i][take_row][take_column]
                dataset[i][take_row][take_column] = 0
                #print('dataset ' + str(i), '\n',dataset[i]) 
            
                        
                  
            else:
                error_MM_container.append(np.max(dataset[i]))
                error_MM_deadlock.append(int(position_1[0] - row_low + 1))
                candidate_smooth = []
                candidate_block = []
                #print('height =', height)
                for column in [x for x in range(0, 6) if x != int(position_1[1]) and height[x] != 4]:
                    for row in range(1, 4):
                        if dataset[i][row][column] > dataset[i][row_low][int(position_1[1])]:
                            candidate_smooth.append(dataset[i][row][column])
                            break
                        
                        elif dataset[i][row][column] < dataset[i][row_low][int(position_1[1])] and dataset[i][row][column] != 0:
                            candidate_block.append(dataset[i][row][column])
                            break
                    for row in range(0, 4):
                        if dataset[i][NTier - height[column] - 1][column] == 0:
                            candidate_smooth.append(NumCon + 1) 
                            zero_row, zero_column = NTier - height[column] - 1, column
                            break
                    break
                        
                #print(candidate_smooth)

                if len(candidate_smooth) == 0:                    
                    con = max(candidate_block)
                    position_con = np.where(dataset[i] == con)
                    dataset[i][int(position_con[0]) - 1][int(position_con[1])] = dataset[i][row_low][int(position_1[1])]
                    last_take = dataset[i][row_low][int(position_1[1])]
                    dataset[i][row_low][int(position_1[1])] = 0
                        
                else:                        
                    con = min(candidate_smooth)
                   
                    if con == NumCon +1:
                        dataset[i][zero_row][zero_column] = dataset[i][row_low][int(position_1[1])]
                        last_take = dataset[i][row_low][int(position_1[1])]
                        dataset[i][row_low][int(position_1[1])] = 0
                    
                    else:
                        position_con = np.where(dataset[i] == con)
                        dataset[i][int(position_con[0]) - 1][int(position_con[1])] = dataset[i][row_low][int(position_1[1])]
                        last_take = dataset[i][row_low][int(position_1[1])]
                        dataset[i][row_low][int(position_1[1])] = 0
                error += 1
                #print('error')
                #print('Round =', Round)
                #print('dataset = ' + str(i), '\n',dataset[i]) 
            
                    
                        
        #take_row, put_row = find_row_top_con(dataset, i, position_1, put_column)
        #dataset[i][put_row - 1][put_column] = dataset[i][take_row][int(position_1[1])]
        #dataset[i][take_row][int(position_1[1])] = 0
        #print('dataset =',dataset[i])
            
        #print('dataset ' + str(i), '\n',dataset[i])       
        
        movement += 1  
        #print('movement =', movement)
        
    movement_1_2 = count_last_two_con(dataset[i])
    movement = movement + movement_1_2
    #print('total movement =', movement)
    #print('movement =',movement)
    dataset_movement.append(movement)
    #print(dataset_movement)
    Round += 1
    #print('Round =', Round)
    #print('error =', error)
#print(dataset_movement)
print('error =', error)
print(sum(dataset_movement))
print(sum(dataset_movement) / len(dataset_movement))           
               
end = time.time()
print(end - start)            


with open('error_MM_container.pickle', 'wb') as file:
    pickle.dump(error_MM_container, file)

with open('error_MM_deadlock.pickle', 'wb') as file:
    pickle.dump(error_MM_deadlock, file) 
   
with open('result_ML_min_max.pickle','wb') as file:
    pickle.dump(dataset_movement, file)

#27066659
#27.066659
#7366.79141330719
#error = 447557
    
# after debug
#26929728
#26.929728
#6172.909444093704
#error = 9088