# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:04:53 2019

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
N = int(2)

with open ('input_layer_4_6_18.pickle', 'rb') as file:
    input_layer_4_6_18 = pickle.load(file)


input_layer_trans = input_layer_4_6_18
input_layer = np.transpose(input_layer_4_6_18)
#input_layer = input_layer[:, 2268:2269]
#print(input_layer.shape)

dataset = np.transpose(input_layer)
dataset.shape = (len(dataset), NTier, NCol)

#print(len(dataset))
#print('dataset =', dataset)
#print(dataset)
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
        caches['A' + str(1)] = sigmoid(caches['Z' + str(1)])
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

def find_row_top_con(dataset, i, position_1, put_column, take_column):
    
    put_row = 4
    for row in range(0, 4):
        if dataset[i][row][take_column] != 0:
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

def Find_Stack_N(Bay, N, lowest_con):
    Stack_N = []
    
    for i in range(lowest_con, lowest_con + N): # i equals tocontainer
        
        con_position = np.where(Bay == i)
        Stack_N.append(int(con_position[1]))
    
    return list(set(Stack_N))

def Find_Top_Stack_N(Bay, Stack_N):
    
    Top_Stack_N = []
    for i in Stack_N:
        for j in range(0, NTier):
            if Bay[j][i] < NumCon + 1 and Bay[j][i] != 0:
                Top_Stack_N.append(Bay[j][i])
                break
    
    return Top_Stack_N

def Find_Lowest_con_in_column(Bay, Stack, high_con):  # Stack = not in stack_N, find the lowest number container in non-Stack_N
    Low_Stack = []
    zero_row = 3
    zero_column = 0
    #print(Stack)
    for i in Stack:
        for j in range(0, NTier):
            '''
            if Bay[NTier - 1][i] == 0:
                Low_Stack.append(high_con + 1)
                zero_row, zero_column = NTier - 1, i
            '''
            if Bay[j][i] > 0:
                Low_Stack.append(Bay[j][i])
            
                break
        
    for i in Stack:
        if Bay[NTier - 1][i] == 0:
            Low_Stack.append(high_con + 1)
            zero_row, zero_column = NTier - 1, i

        
    return Low_Stack, zero_row, zero_column


dataset_movement = []
Round = 0 
error = 0

error_LA_container = []
error_LA_deadlock = []

for i in range(0, len(dataset)):
    movement = 0
    last_take = 0

    #print(np.sum(dataset[i]))
    #print('dataset =' + str(i), '\n',dataset[i])
    while np.sum(dataset[i]) > 3: #Because there will be only con1 and con2 
    
        high_con = np.max(dataset[i]) #find the highest number container in the bay
        position_1 = np.where(dataset[i] == 1) #find the position of container 1
        height = Height(dataset[i]) # show the height of the bay configuration
        
        for row in range(0, 4): # this loop is for finding where the container exists 
            if dataset[i][row][int(position_1[1])] >= 1:
                row_low = row
                break
        #print(high_con, position_1[0] - row_low + 1)
        if int(position_1[0]) - row_low + 1 == 1: # In this if condition, retrieving container 1 if the top container in target stack is container 1.
            if row_low == int(position_1[0]):
                dataset[i][int(position_1[0])][int(position_1[1])] = 0 
                dataset[i] = start_from_one(dataset[i], NTier, NCol)
                #print('dataset =' + str(i), '\n',dataset[i])
                last_take = 1
                
        else:
            with open("weight_LA_N_4_6_" + str(high_con) + '_' + str(int(position_1[0] - row_low + 1)) + '.pickle', 'rb') as file: #find the weight for different type of bay configuration
                Parameters = pickle.load(file)
            L = int(len(Parameters) / 2 + 1)           
            x = input_layer[:, i:i+1]
            #print('x =', x)
            A, cache = forward_propogation(x, L, Parameters)
            #print('A =', A)
            #A = np.where(A>=0.5,1,0)
            
            # the for loop is for interpreting the ML result to [1, 0] form. 1 means the stack where new container are put. 
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
            
            #print('A =', A)       
            #print(len(A))
            #print('dataset =' + str(i), '\n',dataset[i])
            N = min(N, high_con)
            for column in range(0, 6):
                if A[column] == 1:
                    take_column = column  
                    break
            #print('take_column =', take_column)
            
            for column in range(6, 12):
                if A[column] == 1:
                    put_column = column
            put_column = put_column - 6    
            #print('put_column =', put_column)
            
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
            
            
            safe = 0
            while safe < 1:
                
                Stack_N = Find_Stack_N(dataset[i], N, 1)
                Not_Stack_N = [x for x in range(0, 6) if x not in Stack_N and height[x] != 4]                
                if len(Not_Stack_N) == 0:
                    N -= 1
                else:
                    safe += 1
            #print('Stack_N = ', Stack_N)
            #print('Not_Stack_N =', Not_Stack_N)
            Top_Stack_N = Find_Top_Stack_N(dataset[i], Stack_N)
            #print('Top_Stack_N =', Top_Stack_N)
            Low_con_not_Stack_N, zero_row, zero_column = Find_Lowest_con_in_column(dataset[i], Not_Stack_N, high_con)
            #print('zero_row = ', zero_row)
            #print('zero_column =', zero_column)
            #print('Low_con_not_Stack_N = ', Low_con_not_Stack_N)
            
            if dataset[i][0][put_column] == 0 and dataset[i][int(position_1[0])][put_column] != 1 and dataset[i][3][take_column] != 0 and dataset[i][take_row][take_column] != last_take and take_column != put_column:                       
                                       
                dataset[i][put_row - 1][put_column] = dataset[i][take_row][take_column]
                last_take = dataset[i][take_row][take_column]
                dataset[i][take_row][take_column] = 0
                #print('dataset ' + str(i), '\n',dataset[i])
                
            else:
                r = 1
                error_LA_container.append(np.max(dataset[i]))
                error_LA_deadlock.append(int(position_1[0] - row_low + 1))
                while r <= len(Stack_N):
                    n = max(Top_Stack_N)
                    position_n = np.where(dataset[i] == n)
                    #print('n =', n)
                    candidate_smooth = []
                    candidate_block = []
                    for con in Low_con_not_Stack_N:
                        if con > n:
                            candidate_smooth.append(con)
                        else:
                            candidate_block.append(con)
                    #print('candidate_smooth = ', candidate_smooth)
                    #print('candidate_block = ', candidate_block)
                    
                    if r != len(Stack_N):
                        
                        if len(candidate_smooth) > 0:

                            con = min(candidate_smooth)
                            
                            if con == high_con + 1:
                                position_n = np.where(dataset[i] == n)
                                dataset[i][zero_row][zero_column] = n
                                last_take = n
                                dataset[i][int(position_n[0])][int(position_n[1])] = 0
                            else:
                                position_con = np.where(dataset[i] == con)
                                position_n = np.where(dataset[i] == n)
                                dataset[i][int(position_con[0])-1][int(position_con[1])] = n
                                last_take = n
                                dataset[i][int(position_n[0])][int(position_n[1])] = 0
                            error += 1
                            #print('dataset =' + str(i), '\n',dataset[i])
                            #print('error shows')
                            break
                    
                        elif position_n[1] == position_1[1]:

                            if len(candidate_smooth) > 0:
                                con = min(candidate_smooth)

                            else:
                                con = max(candidate_block)
                            position_con = np.where(dataset[i] == con)
                            dataset[i][int(position_con[0])-1][int(position_con[1])] = n
                            last_take = n
                            dataset[i][int(position_n[0])][int(position_n[1])] = 0
                            error += 1
                            #print('error shows')
                            break
                        
                        else:
                            r += 1
                            
                    elif r == len(Stack_N):

                        if len(candidate_smooth) > 0:
                            con = min(candidate_smooth)
                            if con == high_con + 1:
                                position_n = np.where(dataset[i] == n)
                                dataset[i][zero_row][zero_column] = n
                                last_take = n
                                dataset[i][int(position_n[0])][int(position_n[1])] = 0
                                
                            else:                                                                                      
                                position_con = np.where(dataset[i] == con)
                                position_n = np.where(dataset[i] == n)
                                dataset[i][int(position_con[0])-1][int(position_con[1])] = n
                                last_take = n
                                dataset[i][int(position_n[0])][int(position_n[1])] = 0
                            #print('dataset =' + str(i), '\n',dataset[i])
                            error += 1
                            #print('error shows')
                    
                        else:
                            
                            con = max(candidate_block)
                            position_con = np.where(dataset[i] == con)
                            position_n = np.where(dataset[i] == n)
                            dataset[i][int(position_con[0])-1][int(position_con[1])] = n
                            last_take = n
                            dataset[i][int(position_n[0])][int(position_n[1])] = 0
                            error += 1
                            #print('error shows')
                    
                        r += 1
                        
                    Top_Stack_N.remove(max(Top_Stack_N))
                        
        #print('dataset =' + str(i), '\n',dataset[i])
                            
        movement += 1  
        #print('movement =', movement)
        #print('error =', error)
    movement_1_2 = count_last_two_con(dataset[i])
    movement = movement + movement_1_2
    #print('total movement =', movement)
    #print('movement =',movement)
    #print('error =', error)
    dataset_movement.append(movement)
    Round += 1
    #print('Round =', Round)
    #print(dataset_movement)

#print(dataset_movement)
print(sum(dataset_movement))
print(sum(dataset_movement) / len(dataset_movement))
print('error =', error)                

with open('LA_N_ML_error_LA_container.pickle', 'wb') as file:
    pickle.dump(error_LA_container, file)

with open('LA_N_ML_error_LA_deadlock.pickle', 'wb') as file:
    pickle.dump(error_LA_deadlock, file) 


with open('LA_N_ML.pickle', 'wb') as file:
    pickle.dump(dataset_movement, file)
         
end = time.time()
print(end - start)                    
                            
#27009937
#27.009937
#error = 15837
#6341.0865483284
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                