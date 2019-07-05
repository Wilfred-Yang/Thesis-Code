# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 18:30:41 2018

@author: user
"""
# =============================================================================
# The program is for creating input data by combining permutation of labeled container and permutation of column height to generate bay configurations.
# Here are the steps to generate bay configurations: 1. generating all combinations of column height
#                                                    2. generating the permutation of labeled containers
#                                                    3. combinating the first step and the second step to generate bay configurations
#                                                    4. filting bay configurations by checking the number of deadlocks 
# =============================================================================
                   

import itertools
import numpy as np
import random
import pickle
np.set_printoptions(threshold=np.inf) #enable computer to show all arrays on the monitor
'''
NTier = int(input("enter the number of height(NTier) in the bay :"))
NCol = int(input("enter the number of column(NCol) in the bay:"))
NumCon =int(input("enter the number of total containers(NumCon) in the bay:"))
'''
# =============================================================================
# Deciding the size and container of the bay
# =============================================================================
NTier = 4  #Number of rows 
NCol = 6   #Number of columns
NumCon = 3 #Number of containers

# =============================================================================
# The function is for generating the permutation of column height
# =============================================================================
def conbination_height(NTier, NCol, NCon):                                     #(Number of rows , Number of columns, Number of containers)
    Height = []                                                                #Craeting a list for putting all combination of column height
    combination_container = itertools.product(range(NTier + 1), repeat = NCol) #Applying Cartesian product. If row = 4, column = 3, the result will be the collection of triplets ((0,0,0)
                                                                               #, (0,0,1), (0,0,2), (0,0,3), (0,1,0), (0,1,1),...,(4,4,1), (4,4,2), (4,4,3), (4,4,4))
    for i in combination_container:                                            #for each triplet in the collection of Cartesian product
        
        if sum(i) == NCon:                                                     #Summing up the value and append it if the value is equal to the number of containers
            Height.append(i)                                                   #which means the triplet is a possible column height
    return Height                                                              
Height_combination = conbination_height(NTier, NCol, NumCon)                   #Creating the all combinations of column height



num_lists= [i for i in range(1, NumCon + 1)]                                   #Creating a list which contains all labeled containers
permutations_container = itertools.permutations(num_lists)                     #Using the list to create permutation of labeled containers 
#for example, NumCon(Number of container) = 3, the permutation will be (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1). 6 permutations 
#The number represents the labeled containers. When putting containers in the bay, there are different ways to put containers in different permutations.


# =============================================================================
# The function generates bay configurations by applying combinations of column height and permutation of labeled containers
# =============================================================================
def bay_data(Height_combination, permutations_container):          #(combinations of column height, permutation of label containers)

    for each_combination in Height_combination:                    #for each possible column height
        #print('each_combination =', each_combination)
        permutations_container, permutations_container_back_up = itertools.tee(permutations_container) #Creating repeated generator, because one generator can only run loop once. 
        
        # =============================================================================
        # The first for loop has limited the position that where the containers should be. 
        # Thus, the second for loop decide what containers to be put by following different permutation of containers
        # =============================================================================
        
        for each_container_permutations in permutations_container: #set the for loop to put container by following column height in last for loop  
            #print('each_container_permutations = ', each_container_permutations)
            bay = np.zeros((NTier, NCol), dtype = int)             #Creating an empty bay
            order = 0                                              #order is for each permutation's order of putting container, it shows the which container will be put in bay
            
            
            # The for loop below is for putting container in the bay 
            for j in range(0, NCol): 
                #print('j =', j)
                count = each_combination[j]                        #count means how many containers should be put in this column 
                row_position = 3                                   # means row (0 to row - 1, example: 0 to 3, from the biggest number means putting the container from the bottom to the top)
                #print('count =', count)
                while count > 0: 
                    #print('order =', order)
                    bay[row_position][j] = each_container_permutations[order] #order in each permutation of labeled containers
                    #print('bay =', bay)
                    row_position -= 1                              #There is a container in this row. Thus, moving to the higher position
                    count -= 1                                     #When finishing putting a container, the position for container putting in the column reduce 1
                    order += 1                                     #When finishing putting a container, the order turns to the next container
            #print('each_container_permutations = ', each_container_permutations)
            #print('bay = ','\n', bay)
            yield bay
        permutations_container = permutations_container_back_up    #replace repeated generator
    

Data_bay = bay_data(Height_combination, permutations_container)    #creating bay configurations bay by applying all combinations of column height and permutation of label containers  
#print(Data_bay)


# =============================================================================
# The function is to find bay configuration that match the given number of deadlocks
# =============================================================================
def con_one_on_top(): 
    
    for i in Data_bay:                      # for each generated bay configuration
        #print(i)
        for row in range(0, NTier):
            for column in range(0, NCol):
                if i[row][column] == 1:     #find the position of container 1 by applying nested for loop, row and column mean the position of container 1 here

                    count = 0               # counting how many 0(empty position) in this column
                    for row_above_con_one in range(0, row):
                        if i[row_above_con_one][column] == 0:
                            count += 1
                    
                    if count == row - 1:    #adjust the row!!!!!!!! if container one is under a container, turn it to row - 1 
                                            #if count equals to number of row, it means there is no container above container 1 
                        yield i 
                        
con_one_top = con_one_on_top()              #filting bay configurations by checking the number of deadlocks 

Bay_data = []                        
count = 0
# =============================================================================
# The for loop below is append bay configuration into the list 
# =============================================================================
for i in con_one_top:
    print(i)
    count += 1
    #print(count)
    Bay_data.append(i)

Bay_data = np.asarray(Bay_data)                #Turn the list into array

Bay_data.shape = (len(Bay_data), NTier * NCol) #Reshape the size of generated bay configurations that turns to input data to match the ANN model
print(Bay_data)
#print(len(Bay_data)) 
    
with open('input_layer_4_6_3_2.pickle', 'wb') as file: #Saving the generated input data as a pickle file
    pickle.dump(Bay_data, file)
