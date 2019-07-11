# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:08:52 2019

@author: user
"""
# =============================================================================
# The program verifies the ANN-based system. I have randomly generated 1 million different bay configuration to do verification. 
# The size of the bay is 4-row, 6-column, and 18-container.
# The program utilize trained parameters from different type of dataset to reshuffle containers. 
# For example, if the situation of bay is that 17 container are in the bay and there are 2 deadlocks above the target container, the program will apply the trained
# parameters named 4_6_17_3 to reshuffle container once. And the program will start to check the bay situation to find a suitable trained parameters to reshuffle containers
# There have been programs emptying 1 million different bay configuration by Min-Max and Look-ahead N heuristics. 
# When there is an error, the program will check which heuristic performs better in the bay configuration and apply the heuristic to reshuffle container
# Here are the steps to empty the bay:
#     1.Loading the input data and turn them into bay configuration
#     2.Applying the ANN to get output data
#     3.The ANN doesn't deal with container 1 on the top, because it's easy to recognize in real life
#       Thus, the program will directly retrieve container 1 when it's on the top postion. 
#     4.Using output data to reshuffle container
#     5.If there is unreasonable reshuffle, the program will regard it as an error and apply Min-Max or Look-ahead N heuristics to reshuffle the container
#     6.Calculating the average movement (reshuffle + retrievl) on emptying the bay
# =============================================================================

import numpy as np
import pickle
import time
np.set_printoptions(threshold=np.inf)
start = time.time()

NTier = int(4)                     #Number of rows
NCol = int(6)                      #Number of columns
NumCon = int(18)                   #Number of containers
N = int(2)                         #N represents the lowest N number containers in the bay. If there are 5 containers (container 1,2,3,4,5), 
                                   #container 1 and 2 are lowest number containers 

# =============================================================================
# Loading the 1 million bay configurations
# =============================================================================
with open ('input_layer_4_6_18.pickle', 'rb') as file:
    input_layer_4_6_18 = pickle.load(file)

# =============================================================================
# Loading the record on the better performance of heuristic on each bay configuration
# =============================================================================
with open('Min_better_LA_N_4_6_18.pickle', 'rb') as file:
    Min_better_LA_N = pickle.load(file)    
#print(Min_better_LA_N)
#print(len(Min_better_LA_N))
input_layer_trans = input_layer_4_6_18

input_layer = np.transpose(input_layer_4_6_18)  #Turn them as input data. The size changes from (1000000, 24) to (24, 1000000) 
#input_layer = input_layer[:, 12721:12722]
#print(input_layer.shape)

dataset = np.transpose(input_layer)             #The size changes from (24, 1000000) to (1000000, 24)
dataset.shape = (len(dataset), NTier, NCol)     #Turning input data into bay configuration. The size changes from (1000000, 24) to (1000000, 4, 6)
#print(dataset)
#print(dataset.shape)

# =============================================================================
# The function is sigmoid function
# =============================================================================
def sigmoid(Z):
    A = 1/ (1+np.exp(-Z))
    
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
# The function is forward propagation, feel free to reference the file 'ANN_sample' to understand this function
# =============================================================================
def forward_propogation(X, L, Parameters):
    caches = {}
    if L == 2:
        caches['Z'+ str(1)] = np.dot(Parameters['W1'], X) + Parameters['b1']
        caches['A'+str(1)] = softmax(caches['Z'+ str(1)])
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
        caches['A' + str(L-1)] = softmax(caches['Z' + str(L-1)])
        A = caches['A'+str(L-1)]
  

    return A, caches

# =============================================================================
# The function is uncessnary here, feel free to skip it
# =============================================================================
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

# =============================================================================
# Ensuring the lowest number in the bay that is always 1. 
# The ANN learn from the input data from bay configuration, and containers in bay configuration are all from 1 
# =============================================================================
def start_from_one(Bay, NTier, NCol): #let the bay order start from container one
    for row in range(0, NTier):
        for column in range(0, NCol):
            if Bay[row][column] > 0:
                Bay[row][column] = Bay[row][column] - 1
    
    return Bay

# =============================================================================
# The function works when there are only two containers in the bay. 
# The function counts number of movement to empty the two containers.
# =============================================================================
def count_last_two_con(Bay):
    position_2 = np.where(Bay == 2)                       #Find the position of container 2
    position_1 = np.where(Bay == 1)                       #Find the position of container 1
    if int(position_2[1]) == int(position_1[1]):          #if the two containers are in the same column
        if int(position_1[0]) > int(position_2[0]):       #Checking which container is on higher position
            movement_1_2 = 3                              #if container 2 is on higher position, container 2 will be reshuffled first.
                                                          #and container 1 and 2 will be retrieved. The total movement is 3
                                                          
        else:                                             #There is no deadlock. Retrieving 2 containers
            movement_1_2 = 2
    else:                                                 #The two containers are in different column. Retrieving 2 containers          
        movement_1_2 = 2
    
    return movement_1_2

# =============================================================================
# Calculating the column height of the bay
# =============================================================================
def Height(Bay):
    height = np.zeros((6,), dtype = int)
    for row in range(0, 4):
        for column in range(0, 6):
            if Bay[row][column] >= 1:
                height[column] += 1
    
    return height

# =============================================================================
# The function finds columns where the lowest N containers exist. Feel free to reference the program 'Output_Data_LA_N_sample' to check the comments
# =============================================================================
def Find_Stack_N(Bay, N, lowest_con):
    Stack_N = []
    
    for i in range(lowest_con, lowest_con + N): # i equals to container
        
        con_position = np.where(Bay == i)
        Stack_N.append(int(con_position[1]))
    
    return list(set(Stack_N))

# =============================================================================
# The function finds container on the top of the column where the lowest N containers exist. 
# Feel free to reference the program 'Output_Data_LA_N_sample' to check the comments
# =============================================================================
def Find_Top_Stack_N(Bay, Stack_N):
    
    Top_Stack_N = []
    for i in Stack_N:
        for j in range(0, NTier):
            if Bay[j][i] < NumCon + 1 and Bay[j][i] != 0:
                Top_Stack_N.append(Bay[j][i])
                break
    
    return Top_Stack_N

# =============================================================================
# The function finds the lowset number container in the column
# =============================================================================
def Find_Lowest_con_in_column(Bay, Stack, high_con):     #Stack = For columns are not in stack_N here
    Low_Stack = []                                       #For recording the lowest number container in each column of Stack
    zero_row = 3                                         
    zero_column = 0
    #print(Stack)
    for i in Stack:
        con = high_con + 1 
        for j in range(0, NTier):
            if Bay[j][i] < con and Bay[j][i] != 0:
                con = Bay[j][i]
        Low_Stack.append(con)

    # =============================================================================
    # The for loop sets the position of row and column when the column is empty     
    # =============================================================================
    for i in Stack:
        if Bay[NTier - 1][i] == 0:
            #Low_Stack.append(high_con + 1)
            zero_row, zero_column = NTier - 1, i

        
    return Low_Stack, zero_row, zero_column

dataset_movement = []
Round = 0 
error_MM = 0                                                                   #For recording the error that Min-Max heuristic needs to deal 
error_LA = 0                                                                   #For recording the error that Look-ahead N heuristic needs to deal 
error_LA_container = []                                                        #For recording which bay configuration shows the error
error_LA_deadlock = []                                                         #For recording the deadlock when the error shows
error_MM_container = []                                                        #For recording which bay configuration shows the error
error_MM_deadlock = []                                                         #For recording the deadlock when the error shows

# =============================================================================
# The for loop starts to empty containers on each different bay configuration 
# =============================================================================
for i in range(0, len(dataset)):
    movement = 0                                                               #For recording the movement of each bay configuration 
    last_take = 0                                                              #For recording the container that program moved last time
    
    #print('dataset =' + str(i), '\n',dataset[i])
    while np.sum(dataset[i]) > 3:                                              #If there are only container 1 and 2 in the bay, the while loop will stop 
            
        high_con = np.max(dataset[i])                                          #Highset number container   
        position_1 = np.where(dataset[i] == 1)                                 #Finding the position of container 1
        height = Height(dataset[i])                                            #Calculating the column height of the bay
        
        # =============================================================================
        # The for loop finds row position of the top container in the column where container 1 locates 
        # =============================================================================
        #print(dataset[i])
        for row in range(0, 4):
            if dataset[i][row][int(position_1[1])] >= 1:
                row_low = row
                break
        # =============================================================================
        # The if condition checks if the container 1 is on the top postion. If yes, retrieving container 1. If no, reshuffling container in else condition
        # =============================================================================            
        if int(position_1[0]) - row_low + 1 == 1:
            if row_low == int(position_1[0]):
                dataset[i][int(position_1[0])][int(position_1[1])] = 0 
                dataset[i] = start_from_one(dataset[i], NTier, NCol)
                #print('dataset =' + str(i), '\n',dataset[i])
                last_take = 1
        
        else:                
            with open("weight_combine_4_6_" + str(high_con) + '_' + str(int(position_1[0] - row_low + 1)) + '.pickle', 'rb') as file: #Loading the file of trained parameters
                Parameters = pickle.load(file)
            L = int(len(Parameters) / 2 + 1)                                   #Calculating number of layers
            x = input_layer[:, i:i+1]                                          #Load each bay configuration
            #print('x =', x)
            A, cache = forward_propogation(x, L, Parameters)                   #Getting predicted A from forward propagation
            
            # =============================================================================
            # The following two for loop turn the predicted A into binary integer. The size of predicted A is (12, 1)         
            # =============================================================================                
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
            # =============================================================================
            # The following two for loop find the columns that where the container is taken and put 
            # =============================================================================            
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
            
            # =============================================================================
            # The following two for loop and one if condition find the rows that where the container is taken and put
            # =============================================================================            
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
                
            # =============================================================================
            # If Look-ahead N performs better than Min-Max in the bay configuration, the error will be reshuffled by Look-ahead N heuristic 
            # =============================================================================
            if i not in Min_better_LA_N:
                
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
                
                
                #Finding the lowest number container in the column of Not_Stack_N and get the position of empty column (row, column)
                #If there are many empty columns, the program will fix a column to reshuffle container which are 'zero_row' and 'zero_column'                 
                Low_con_not_Stack_N, zero_row, zero_column = Find_Lowest_con_in_column(dataset[i], Not_Stack_N, high_con)
                
                
                #print('zero_row = ', zero_row)
                #print('zero_column =', zero_column)
                #print('Low_con_not_Stack_N = ', Low_con_not_Stack_N)
                
                #print('Last_take =', last_take)
                #print('Take_row =', take_row)
                #print('Take_column =', take_column)
                #print('Put_row =', put_row)
                #print('Put_column =', put_column)
                 # =============================================================================
                 # Because the ANN dosen't always make a reasonable decision, there are some settings to avoid error in the process of reshuffling containers.
                 # Here are some situations that cause an error that the program is unable to reshuffle containers:  
                 #     1. When there is no empty position but the result of ANN shows that container should be reshuffled that column 
                 #     2. ANN decides to put container into the column where the container 1 exists
                 #     3. ANN decides to take container from the column where no container exists
                 #     4. ANN repeatly reshuffles the same container.
                 #     5. ANN reshuffles container to the same column
                 # If the situations above don't show, the program will apply the result of ANN to reshuffle the container
                 # Else, Min-Max heuristic is applied to reshuffle the container
                 # =============================================================================
            

                if dataset[i][0][put_column] == 0 and dataset[i][int(position_1[0])][put_column] != 1 and dataset[i][3][take_column] != 0 and dataset[i][take_row][take_column] != last_take and take_column != put_column:                       
                                       
                    dataset[i][put_row - 1][put_column] = dataset[i][take_row][take_column]
                    last_take = dataset[i][take_row][take_column]
                    dataset[i][take_row][take_column] = 0
                    #print('dataset ' + str(i), '\n',dataset[i])
                    
                
                else:
                    r = 1
                    while r <= len(Stack_N):
                        n = max(Top_Stack_N)                                            #Testing to reshuffle container n
                        position_n = np.where(dataset[i] == n)                          #Finding the position (row, column) of n
                        #print('n =', n)
                        candidate_smooth = []                                           #For recording the lowest number container in each column of Low_con_not_Stack_N that there is no new deadlock after reshuffle
                        candidate_block = []                                            #For recording the lowest number container in each column of Low_con_not_Stack_N that there is a new deadlock after reshuffle
                        # =============================================================================
                        # The for loop records container in list 'candidate_smooth' and 'candidate_block'
                        # =============================================================================
                        for con in Low_con_not_Stack_N:
                            if con > n:
                                candidate_smooth.append(con)
                            else:
                                candidate_block.append(con)
                        #print('candidate_smooth = ', candidate_smooth)
                        #print('candidate_block = ', candidate_block)
                        # =============================================================================
                        # If the order of container in Top_Stack_N isn't the last                      
                        # =============================================================================
                        if r != len(Stack_N):
                            error_LA_container.append(np.max(dataset[i]))                  #Record the container of the error, feel free to skip it
                            error_LA_deadlock.append(int(position_1[0] - row_low + 1))     #Record the deadlock of the error, feel free to skip it
                            
                            # =============================================================================
                            # If reshuffling container n doesn't cause a new deadlock, reshuffling it
                            # =============================================================================
                            if len(candidate_smooth) > 0:
                                con = min(candidate_smooth)                                #Picking the smallest number container called con
                                
                                # =============================================================================
                                # If the there are no container in the column where 'candidate_smooth' locates, reshuffling container n to the empty column                 
                                # =============================================================================
                                if con == high_con + 1:                                   
                                    position_n = np.where(dataset[i] == n)
                                    dataset[i][zero_row][zero_column] = n
                                    last_take = n
                                    dataset[i][int(position_n[0])][int(position_n[1])] = 0
                                # =============================================================================
                                # Else; reshuffling container n to the column where con locates                                
                                # =============================================================================
                                else:
                                    position_con = np.where(dataset[i] == con)
                                    position_n = np.where(dataset[i] == n)
                                    
                                   # =============================================================================
                                   # The for loop finds the position which container n reshuffle to  
                                   # =============================================================================                                                                        
                                    for row in range(0, NTier):
                                    
                                        if dataset[i][row][int(position_con[1])] > 0:
                                            dataset[i][row-1][int(position_con[1])] = n
                                            break                                    
                                    #dataset[i][int(position_con[0])-1][int(position_con[1])] = n
                                    last_take = n
                                    dataset[i][int(position_n[0])][int(position_n[1])] = 0
                                    
                                error_LA += 1                                             #Recording the error
                                #print('dataset =' + str(i), '\n',dataset[i])
                                #print('error shows')
                                break
                            # =============================================================================
                            # If container n and container 1 locate in the same column, the program will immidiately reshuffle container n        
                            # =============================================================================
                            elif position_n[1] == position_1[1]:
                                
                                # =============================================================================
                                # If reshuffling container n doesn't create a new deadlock, reshuffling container n to the column where the smallest number container in 'candidate_smooth' locates
                                # Else; reshuffling container n to the column where the largest number container in 'candidate_block' locates
                                # =============================================================================
                                if len(candidate_smooth) > 0:
                                    con = min(candidate_smooth)
                                    
                                else:
                                    con = max(candidate_block)
                                                                
                                position_con = np.where(dataset[i] == con)                #Finding the position(row, column) of con
                                
                                # =============================================================================
                                # The for loop finds the position which container n reshuffle to  
                                # =============================================================================
                                for row in range(0, NTier):
                                    
                                    if dataset[i][row][int(position_con[1])] > 0:
                                        dataset[i][row-1][int(position_con[1])] = n
                                        break
                                #dataset[i][int(position_con[0])-1][int(position_con[1])] = n
                                last_take = n                                             
                                dataset[i][int(position_n[0])][int(position_n[1])] = 0
                                error_LA += 1
                                #print('dataset =' + str(i), '\n',dataset[i])
                                #print('error shows')
                                
                                break
                            
                            else:
                                r += 1                                                    #Going to the next container in Top_Stack_N
                            
                        # =============================================================================
                        # If it's the last container in Top_Stack_N, the program will reshuffle the container whether there is new deadlock or not  
                        # =============================================================================
                        elif r == len(Stack_N):
                            error_LA_container.append(np.max(dataset[i]))
                            error_LA_deadlock.append(int(position_1[0] - row_low + 1))
                            #print('dataset =' + str(i), '\n',dataset[i])
                            
                            # =============================================================================
                            # If reshuffling container n doesn't cause a new deadlock                           
                            # =============================================================================
                            if len(candidate_smooth) > 0:
                                con = min(candidate_smooth)
                                
                                # =============================================================================
                                # If the there are no container in the column where 'candidate_smooth' locates, reshuffling container n to the empty column  
                                # =============================================================================
                                if con == high_con + 1:
                                    position_n = np.where(dataset[i] == n)
                                    dataset[i][zero_row][zero_column] = n
                                    last_take = n
                                    dataset[i][int(position_n[0])][int(position_n[1])] = 0
                                # =============================================================================
                                # Else; reshuffling container n to the column where con locates                                  
                                # =============================================================================
                                else:                                                                                      
                                    position_con = np.where(dataset[i] == con)
                                    position_n = np.where(dataset[i] == n)
                                    # =============================================================================
                                    # The for loop finds the position which container n reshuffle to
                                    # =============================================================================                                     
                                    for row in range(0, NTier):
                                        if dataset[i][row][int(position_con[1])] > 0:
                                            dataset[i][row - 1][int(position_con[1])] = n
                                            break
                                    #dataset[i][int(position_con[0])-1][int(position_con[1])] = n
                                    last_take = n
                                    dataset[i][int(position_n[0])][int(position_n[1])] = 0
                                #print('dataset =' + str(i), '\n',dataset[i])
                                error_LA += 1
                                #print('error shows')
                            
                            # =============================================================================
                            # If reshuffling container n causes a new deadlock                              
                            # =============================================================================
                            else:
                            
                                con = max(candidate_block)
                                position_con = np.where(dataset[i] == con)
                                position_n = np.where(dataset[i] == n)
                                
                                # =============================================================================
                                # The for loop finds the position which container n reshuffle to
                                # =============================================================================                                
                                for row in range(0, NTier):
                                    if dataset[i][row][int(position_con[1])] > 0:
                                        dataset[i][row - 1][int(position_con[1])] = n
                                        break
                                #dataset[i][int(position_con[0])-1][int(position_con[1])] = n
                                last_take = n
                                dataset[i][int(position_n[0])][int(position_n[1])] = 0
                                error_LA += 1
                                #print('error shows')
                            #print('dataset =' + str(i), '\n',dataset[i])
                            r += 1
                        
                        Top_Stack_N.remove(max(Top_Stack_N))                               #Remove container n out of the Top_Stack_N
                        
            # =============================================================================
            # If Min-Max performs better than Min-Max in the bay configuration, the error will be reshuffled by Min-Max heuristic  
            # =============================================================================
            elif i in Min_better_LA_N:
                
                if dataset[i][0][put_column] == 0 and dataset[i][int(position_1[0])][put_column] != 1 and dataset[i][3][take_column] != 0 and dataset[i][take_row][take_column] != last_take and take_column != put_column:                       
                                        
                    dataset[i][put_row - 1][put_column] = dataset[i][take_row][take_column]
                    last_take = dataset[i][take_row][take_column]
                    dataset[i][take_row][take_column] = 0
                    #print('dataset ' + str(i), '\n',dataset[i])  
                
                else:
                    error_MM_container.append(np.max(dataset[i]))                            #Record the container of the error, feel free to skip it
                    error_MM_deadlock.append(int(position_1[0] - row_low + 1))               #Record the deadlock of the error, feel free to skip it
                    
                    candidate_smooth = []  #For recording the lowest number container in each column of Low_con_not_Stack_N that there is no new deadlock after reshuffle
                    candidate_block = []   #For recording the lowest number container in each column of Low_con_not_Stack_N that there is a new deadlock after reshuffle                  
                    
                    Not_Stack_N = [x for x in range(0, 6) if x != int(position_1[1]) and height[x] != 4]    #Finding the columns that container 1 doesn't exist and the columns aren't full                    
                    
                    #Finding the lowest number container in the column of Not_Stack_N and get the position of empty column (row, column)
                    #If there are many empty columns, the program will fix a column to reshuffle container which are 'zero_row' and 'zero_column' 
                    Low_con_not_Stack_N, zero_row, zero_column = Find_Lowest_con_in_column(dataset[i], Not_Stack_N, high_con) 
                    
                    # =============================================================================
                    # The for loop finds the position which container n reshuffle to
                    # =============================================================================
                    for row in range(0, 4):
                    
                        if dataset[i][row][int(position_1[1])] > 0:
                            n = dataset[i][row][int(position_1[1])]
                            break
                        
                    #print('n =', n)
                    # =============================================================================
                    # The for loop records container in list 'candidate_smooth' and 'candidate_block'
                    # =============================================================================
                    for con in Low_con_not_Stack_N:
                        if con > n:
                            candidate_smooth.append(con)
                        else:
                            candidate_block.append(con)
                    # =============================================================================
                    # If reshuffling container n causes a new deadlock                 
                    # =============================================================================
                    if len(candidate_smooth) == 0:                    
                        con = max(candidate_block)
                        position_con = np.where(dataset[i] == con)
                        
                        # =============================================================================
                        # The for loop finds the position which container n reshuffle to
                        # =============================================================================                        
                        for row in range(0, NTier):                                    
                            if dataset[i][row][int(position_con[1])] > 0:
                                dataset[i][row-1][int(position_con[1])] = n
                                break
                        #dataset[i][int(position_con[0]) - 1][int(position_con[1])] = dataset[i][row_low][int(position_1[1])]
                        #last_take = dataset[i][row_low][int(position_1[1])]
                        last_take = n
                        dataset[i][row_low][int(position_1[1])] = 0
                        
                    # =============================================================================
                    # If reshuffling container n doesn't cause a new deadlock                 
                    # =============================================================================                    
                    else:                        
                        con = min(candidate_smooth)
                        
                        # =============================================================================
                        # If the there are no container in the column where 'candidate_smooth' locates, reshuffling container n to the empty column  
                        # =============================================================================                        
                        if con == high_con +1:
                            dataset[i][zero_row][zero_column] = dataset[i][row_low][int(position_1[1])]
                            last_take = dataset[i][row_low][int(position_1[1])]
                            dataset[i][row_low][int(position_1[1])] = 0
                            
                        # =============================================================================
                        # Else; reshuffling container n to the column where con locates                                  
                        # =============================================================================
                        else:
                            position_con = np.where(dataset[i] == con)
                            print('dataset = ' + str(i), '\n',dataset[i])
                            
                            # =============================================================================
                            # The for loop finds the position which container n reshuffle to
                            # =============================================================================                              
                            for row in range(0, NTier):
                                if dataset[i][row][int(position_con[1])] > 0:
                                    dataset[i][row - 1][int(position_con[1])] = n
                                    break                            
                            #dataset[i][int(position_con[0]) - 1][int(position_con[1])] = dataset[i][row_low][int(position_1[1])]
                            #last_take = dataset[i][row_low][int(position_1[1])]
                            last_take = n
                            dataset[i][row_low][int(position_1[1])] = 0
                        
                    error_MM += 1
                    #print('error')
                    #print(error_MM)
                    #print('Round =', Round)
                    #print('dataset = ' + str(i), '\n',dataset[i]) 
                
        #print('dataset =' + str(i), '\n',dataset[i])        
        movement += 1 
        
    movement_1_2 = count_last_two_con(dataset[i])
    movement = movement + movement_1_2
    #print('total movement =', movement)
    #print('movement =',movement)
    dataset_movement.append(movement)
    #print(dataset_movement)
    Round += 1
    print('Round =', Round)
    
#print(dataset_movement)
print('error_MM =', error_MM)
print('error_LA =', error_LA)
print(sum(dataset_movement))
print(sum(dataset_movement) / len(dataset_movement))           
 
'''      
print(error_LA_container)
print(error_LA_deadlock)
print(error_MM_container)
print(error_MM_deadlock)
'''

with open('combine_ML_error_LA_container.pickle', 'wb') as file:
    pickle.dump(error_LA_container, file)

with open('combine_ML_error_LA_deadlock.pickle', 'wb') as file:
    pickle.dump(error_LA_deadlock, file)

with open('combine_ML_error_MM_container.pickle', 'wb') as file:
    pickle.dump(error_MM_container, file)
    
with open('combine_ML_error_MM_deadlock.pickle', 'wb') as file:
    pickle.dump(error_MM_deadlock, file)


with open('combined_ML.pickle', 'wb') as file:
    pickle.dump(dataset_movement, file)



end = time.time()
print(end - start)

                
#error_MM = 600 
#error_LA = 11370
#26967150
#26.96715
#10156.511909723282
        
#error_MM = 599
#error_LA = 11361
#26966612
#26.966612
#17271.925642490387






























































