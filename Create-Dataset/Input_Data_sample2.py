# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:36:15 2018

@author: user
"""

# =============================================================================
# The program is a sample for creating input data by random generating bay configurations. 
# 
# Here are the steps:  1. Generating empty bay 
#                      2. Setting containers by given labels
#                      3. Randomly putting container in the bay
#                      4. Filtting the generated container bay by observing the number of deadlock on the bay  
# Above is a process for generating a bay configuration. There is a while loop repeating the process until the given number of bay configurations is generated. 
# =============================================================================
                     

import numpy as np
import random
import pickle
import time
#NTier = int(input("enter the number of height(NTier) in the bay :"))
#NCol = int(input("enter the number of column(NCol) in the bay:"))
#NumCon =int(input("enter the number of total containers(NumCon) in the bay:"))

start = time.time()

# =============================================================================
# Deciding the size and container of the bay
# =============================================================================
NTier = int(4)  #Number of rows 
NCol = int(6)   #Number of columns
NumCon = int(8) #Number of containers 

# =============================================================================
# The function is for generating an empty bay
# =============================================================================
def bay_space(nt, nc):                     #(Number of rows , Number of columns)
    return np.zeros((nt, nc), dtype = int) #return an empty bay

# =============================================================================
# The function is for setting containers. The function not only sets containers, but also random the sequence of containers.
# =============================================================================
def rand_init_array(NumC):                 #(Number of containers)
    NumC = list(range(1,NumC+1))           #Setting containers and put them in a list
    return random.sample(NumC, NumCon)     #Random the sequence of container(number) in the list

# =============================================================================
# The function is for randomly putting containers in the empty bay.
# =============================================================================
def put_container_into_the_bay(nt, nc, NumC, RL, bay): #(Number of rows, Number of columns, Number of containers, the list with random sequence of container, empty bay)
    Height = np.zeros((nc,), dtype = int)              #Initializing the height of each column in the empty bay
    for i in range(0, NumC):                           #Setting a for loop to ensure every container in list is allocated a position
        column = random.randint(0, nc-1)               #Randomly choosing a column
        while Height[column] >= nt:                    #The while loop is for ensuring the chosen column isn't full
            column = random.randint(0, nc-1)           #If the column is full, randomly choosing a column again and repeating the while loop
        bay[Height[column]][column] = RL[i]            #Picking the first container in the list to put on the destined column
        Height[column] = Height[column] + 1            #There is a container on the destined column, thus the height needs to plus 1 
    return bay, Height                                 #Return the bay and the height of the bay

# =============================================================================
# Because the order for putting container in the bay is from up to down, It doesn't match the real world for putting containers. (From down to up) 
# Thus, the function inverts the generated bay.
# =============================================================================
def bay_reverse(nt,nc,bay):                   #(number of row, number of column, the generated bay)
    reverse_bay = []                          #Creating a list for inverting the bay
    for i in range(0, nt):                    #Setting a for loop to put every row in the list reverse_bay (From the last row to the first row)
        reverse_bay.append(bay[-i-1])
    reverse_bay = np.asarray(reverse_bay)     #transport the list as one array
    reverse_bay = reverse_bay.reshape(nt, nc) #reshape it as(nt, nc) array
    return reverse_bay
    
# =============================================================================
# The function is for finding number of deadlock in the bay
# =============================================================================
def con_one_position(bay):                    #(the inverted bay)
    position_1 = np.where(bay == 1)           #Finding the position of container 1 (row and column)
    block = 0                                 #Setting the deadlock = 0
    for row in range(0, int(position_1[0])):  #Giving the column of container 1, and using for loop to find number of containers on the container 1 
        if bay[row][int(position_1[1])] > 0:  #If there is a container on the container 1, the deadlock plus 1
            block += 1
    return block                              #Return the number of deadlocks
            
Number_data = 0                               #Setting the argument on the number of bay configutations

dataset = []                                  #Setting a list for putting all generated bay configurations

# =============================================================================
# The while loop is for generating the given number of different bay configurations
# =============================================================================
while Number_data < 205000:                            #Setting the number of bay configurations in the while loop
    
    Random_list = rand_init_array(NumCon)              #Setting labeled containers
    Bay = bay_space(NTier, NCol)                       #Creating an empty bay 
    Bay_Instance, Height = put_container_into_the_bay(NTier, NCol, NumCon, Random_list, Bay) #Randomly putting container in the empty bay
    Initial_Bay = bay_reverse(NTier,NCol,Bay_Instance) #Inverting the generated bay 
    block = con_one_position(Initial_Bay)              #Find the number of deadlocks in the bay
    if block == 1:                         #Giving the number of deadlocks and setting a if condition to check whether there are given number of deadlock in the bay
        Initial_Bay = Initial_Bay.tolist() #Turn array to list into execute the next if condition
 
        if Initial_Bay not in dataset:     #This if condition is for checking whether the bay configuration is repeatedly generated
            
            dataset.append(Initial_Bay)    #Putting the bay configuration in the list dataset
            Number_data += 1
    #print('Number_data =', Number_data)
    

dataset = np.asarray(dataset)                #Turn the list into array
print(dataset)
dataset.shape = (len(dataset), NTier * NCol) #Reshape the size of generated bay configurations to match the ANN model
#print(dataset)
#dataset.shape = (len(dataset), NTier, NCol)
#dataset = np.transpose(dataset)
#print(dataset)

with open('input_layer_4_6_8_2.pickle', 'wb') as file: #Saving the generated bay configurations as a pickle file
    pickle.dump(dataset, file)

stop = time.time()
print('Computational time =',(stop - start))