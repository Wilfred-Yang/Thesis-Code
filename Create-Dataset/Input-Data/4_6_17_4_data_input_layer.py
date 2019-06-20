# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 20:02:14 2018

@author: user
"""

import numpy as np
import random
import pickle
import time
#NTier = int(input("enter the number of height(NTier) in the bay :"))
#NCol = int(input("enter the number of column(NCol) in the bay:"))
#NumCon =int(input("enter the number of total containers(NumCon) in the bay:"))

start = time.time()
NTier = int(4)
NCol = int(6)
NumCon = int(17)

def bay_space(nt, nc):
    return np.zeros((nt, nc), dtype = int)

def rand_init_array(NumC):
    NumC = list(range(1,NumC+1))
    return random.sample(NumC, NumCon)

def put_container_into_the_bay(nt, nc, NumC, RL, bay):
    Height = np.zeros((nc,), dtype = int)
    for i in range(0, NumC):
        column = random.randint(0, nc-1) 
        while Height[column] >= nt:
            column = random.randint(0, nc-1)
        bay[Height[column]][column] = RL[i]
        Height[column] = Height[column] + 1
    return bay, Height


def bay_reverse(nt,nc,bay):
    reverse_bay = []
    for i in range(0, nt):        
        reverse_bay.append(bay[-i-1])    
    reverse_bay = np.asarray(reverse_bay) #transport it as one array
    reverse_bay = reverse_bay.reshape(nt, nc) #reshape it as(nt, nc) array
    return reverse_bay
    

def con_one_position(bay):
    position_1 = np.where(bay == 1)
    block = 0
    for row in range(0, int(position_1[0])):
        if bay[row][int(position_1[1])] > 0:
            block += 1
    return block
            
Number_data = 0 

dataset = []
while Number_data < 105000:
    
    Random_list = rand_init_array(NumCon)
    Bay = bay_space(NTier, NCol)
    Bay_Instance, Height = put_container_into_the_bay(NTier, NCol, NumCon, Random_list, Bay)
    Initial_Bay = bay_reverse(NTier,NCol,Bay_Instance)
    block = con_one_position(Initial_Bay)
    if block == 3:
        Initial_Bay = Initial_Bay.tolist()
 
        if Initial_Bay not in dataset:
            
            dataset.append(Initial_Bay)
            Number_data += 1
            print('Number_data =', Number_data)
    

dataset = np.asarray(dataset)
print(dataset)
dataset.shape = (len(dataset), NTier * NCol)
#print(dataset)
#dataset.shape = (len(dataset), NTier, NCol)
#dataset = np.transpose(dataset)
#print(dataset)

with open('input_layer_4_6_17_4.pickle', 'wb') as file:
    pickle.dump(dataset, file)

stop = time.time()
print('Computational time =',(stop - start))