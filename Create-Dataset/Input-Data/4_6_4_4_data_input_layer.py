# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:21:35 2018

@author: user
"""

import itertools
import numpy as np
import random
import pickle
np.set_printoptions(threshold=np.inf)
'''
NTier = int(input("enter the number of height(NTier) in the bay :"))
NCol = int(input("enter the number of column(NCol) in the bay:"))
NumCon =int(input("enter the number of total containers(NumCon) in the bay:"))
'''
NTier = 4
NCol = 6
NumCon = 4

def conbination_height(NTier, NCol, NCon):
    Height = []
    combination_container = itertools.product(range(NTier + 1), repeat = NCol)
    for i in combination_container:
        if sum(i) == NCon:
            Height.append(i)        
    return Height
Height_combination = conbination_height(NTier, NCol, NumCon)
# above is the process of create all combinations of Height in the bay


num_lists= [i for i in range(1, NumCon + 1)]
permutations_container = itertools.permutations(num_lists)
# above is to create all combinations of container order 

def bay_data(Height_combination, permutations_container):

    for each_combination in Height_combination:
        #print('each_combination =', each_combination)
        permutations_container, permutations_container_back_up = itertools.tee(permutations_container) 
        # the above line is to create repeated generator, because one generator can only run loop once. 
        
        for each_container_permutations in permutations_container:
            #print('each_container_permutations = ', each_container_permutations)
            bay = np.zeros((NTier, NCol), dtype = int) # empty bay
            order = 0 #order is for each permutation's order, it shows the which container will be put in bay
            
            for j in range(0, NCol): 
                #print('j =', j)
                count = each_combination[j] #count means how many containers should be put in this column 
                row_position = 3 # means row (0 to row - 1, example: 0 to 3, from the biggest number means putting the container from the bottom)
                #print('count =', count)
                while count > 0:
                    #print('order =', order)
                    bay[row_position][j] = each_container_permutations[order] #order
                    #print('bay =', bay)
                    row_position -= 1
                    count -= 1
                    order += 1
            #print('each_container_permutations = ', each_container_permutations)
            #print('bay = ','\n', bay)
            yield bay
        permutations_container = permutations_container_back_up # replace repeated generator
    

Data_bay = bay_data(Height_combination, permutations_container) # creating (NTier, NCol) size bay dataset 
#print(Data_bay)


def con_one_on_top(): #The meaning of the generator is to seperate data according to the position of  container 1 
    
    for i in Data_bay: 
        #print(i)
        for row in range(0, NTier):  
            for column in range(0, NCol):
                if i[row][column] == 1: #find the position of container 1 
                    
                    count = 0 # counting how many 0(empty position) in this column
                    for row_above_con_one in range(0, row):
                        if i[row_above_con_one][column] == 0:
                                count += 1
                    
                    if count == row - 3: ## adjust the row!!!!!!!! if container one is under a container, turn it to row - 1 
                            #if count equals to number of row, it means there is no container above container 1 
                            yield i 
                        
con_one_top = con_one_on_top()

Bay_data = []        
count = 0
for i in con_one_top:
    print(i)
    count += 1
    #print(count)
    Bay_data.append(i)


Bay_data = np.asarray(Bay_data)

Bay_data.shape = (len(Bay_data), NTier * NCol) 
#print(Bay_data)
print(len(Bay_data)) 
     
with open('input_layer_4_6_4_4.pickle', 'wb') as file:
    pickle.dump(Bay_data, file)