
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 18:31:53 2018

@author: user
"""
# =============================================================================
# The program apply the Min-Max heuristic to create output data that applies to ANN model.
# To create output data, the program read the file of input data (bunch of bay configurations) and reshuffle them once. 
# The moving pattern of reshuffle is turned into output data. The size of each output data is number of columns * 2. They are binary integer in output data.
# The first half of output data shows which column container A is taken. The second half of output data shows which container A is put.
# Taking a 4-row and 3-column container bay as example. If container is reshuffled from column 1 to column 2, the output data will be [1,0,0,0,1,0]. 
# The first half shows column 1 (1,0,0) and the second half shows column 2 (0,1,0). The result of combining them is [1,0,0,0,1,0].
# The term Column = Stack in my study 
# 
# Here are the steps to create output data:
#     1. Setting a for loop to run each bay configuration, in this for loop:
#         
#        (1) Reading the file of input data (bay configuration)
#        (2) Counting the column height of bay configuration
#        (3) Applying the Min-Max heuristic to do reshuffle once and update the column height of bay configuration
#        
#     2. Calculate the difference between the column height before reshuffling and column height after reshuffling
#     3. Turning the difference in step 2 into output data
# =============================================================================

# =============================================================================
# The concept of the Min-Max heuristic won't be explained here. 
# If you are highly interested in the Min-Max heuristic, free feel to reference the study "A chain heuristic for the Blocks Relocation Problem"
# =============================================================================


import itertools
import numpy as np
import random
import pickle
np.set_printoptions(threshold=np.inf)

# =============================================================================
#Deciding the size and container of the bay
# =============================================================================
NTier = int(4)  #Number of rows 
NCol = int(6)   #Number of columns
NumCon = int(3) #Number of containers


with open('input_layer_4_6_3_2.pickle', 'rb') as file:     #Loading the file of input data 
    Bay = pickle.load(file)

Bay.shape = (len(Bay), NTier, NCol)                        #Change the shape of input data to trun them into bay configurations
#print(Bay)

# =============================================================================
# The function is for turning 0 (empty space) into higher number (Numc + 1) that enables the Min-Max heuristic to do reshuffle in Min-Max heuristic
# =============================================================================
def bay_test(Initial_Bay,NumC):
    Bay_test = Initial_Bay
    np.place(Initial_Bay, Initial_Bay == 0, NumC + 1)      # Turning 0 in the bay into NumCon + 1
    return Bay_test

# =============================================================================
# The function is for reshuffling bay configuration once by the Min-Max heuristic. Though the heuristic is not eaplained, the comments are added.
# =============================================================================
def Min_Max(nt,nc,NumC,Bay,Height):                        #(Number of row, Number of column, Number of container, Bay configuration, Column height of the bay configuration)                 
    it = 0                                                 
    #Movement = 0
    #relocation = 0    
    # =============================================================================
    # The Min-Max heuristic will empty the bay. Thus, the while loop is for emptying every container in the bay    
    # =============================================================================
    while it < NumC:
        p_l_c = np.where(Bay == Bay.min(keepdims = True))  #p_l_c = The position of lowest container (container 1 here)
        '''
        print(Bay)
        
        print(p_l_c)
        '''
        p_r = int(p_l_c[0])                                #the row of p_l_c
        #print('p_r =', p_r)
        p_c = int(p_l_c[1])                                #the column of p_l_c
        # =============================================================================
        # The if condition is useless here, because there is no need to retrieve container in this program. To ensure the complete Min-Max heuristic, I don't delete this if condition      
        # =============================================================================
        if  p_r == nt - int(Height[p_c]):                  #if target container (container 1 here) is on the top of a stack, retrieving (taking out of the bay) it
            Bay[p_r][p_c] = NumC + 1                       #the following two lines are for retrieving process. Turning container 1 into empty (Numc + 1 here)
            Height[p_c] = Height[p_c] - 1                  #The column height reduce one due to a container is retrieved
            '''
            print('Height =', Height) 
            '''
            #Movement += 1
            np.place(Bay, Bay == NumC +1, 0)               #Turning the empty position into 0 for visual checking 
            '''
            print('Round =',Movement,'\n',Bay)
            '''
            np.place(Bay, Bay == 0, NumC + 1)              #Turning the empty position into NumC + 1 to keep running the program 
            '''
            print('\n')
            '''
        elif p_r > nt - Height[p_c]:                       #if target container is not on the top position 
            r = nt - Height[p_c]                           #Setting a value for checking how many deadlocks
            '''
            print('r = ', r)
            '''

            while p_r > r:                                 #Setting a while to reshuffle containers above the target container 
                i = 0
                Height_m = []                              #Creating an empty list for recording the column with full containers                
                c_s_i = Bay.min(axis = 0) - Bay[r][p_c]    #Finding the lowest number container in each column - deadlock (Compare the difference between the lowest number container and deadlock)  
                
                # =============================================================================
                # #The while loop is for recording the column with full containers                 
                # =============================================================================
                while nc > i:                              
                    if Height[i] == nt:
                        Height_m.append(i)                 #Recording column with full containers to list 
                    i = i + 1
                Height_m.append(p_c)                       #Recording the target column to list as well                   
                c_s = np.delete(c_s_i, Height_m, None)     #Selecting candidate of column for reshuffling deadlock to 
                
                # =============================================================================
                # Selecting a best candidate of column in this if-else condition          
                # =============================================================================
                if np.max(c_s) > 0:                        #Finding arg number
                    arg_c = min(i for i in c_s if i > 0)
                else:
                    arg_c = max(c_s)
                    
                l_arg_c = []
                for i in range(0,nc):                      
                    if c_s_i[i] == arg_c:                  #Finding the column location of arg_c
                        l_arg_c.append(i)                  #Recording it to list
                r_arg_c = random.choice(l_arg_c)           #Randomly choosing the column if there are multiple arg_c
                
                Bay[nt-Height[r_arg_c]-1][r_arg_c] = Bay[r][p_c] #Reshuffling containers utilizing r_arg_c to find the row that container should be put on
                Bay[r][p_c] = NumC +1                            #Container is reshuffled, now the original position is empty
                Height[p_c] = Height[p_c] - 1                    #The following two lines adjust the column height this bay
                Height[r_arg_c] = Height[r_arg_c] + 1
                '''
                print('\n')
                print('Height =', Height)
                '''
                #relocation += 1
                #Movement += 1
                np.place(Bay, Bay == NumC +1, 0)           #Turning the empty position into 0 for visual checking 
                '''
                print('Round =',Movement,'\n',Bay)
                '''
                np.place(Bay, Bay == 0, NumC + 1)          #Turning the empty position into NumC + 1 to keep running the program 
                '''
                print('relocation =', relocation)
                print('\n') 
                '''
                r = r + 100
            '''
            Bay[p_r][p_c] = NumC + 1
            Height[p_c] = Height[p_c] - 1
            #print('Height =', Height)
            Movement += 1
            '''
            #print('Height =', Height)
            np.place(Bay, Bay == NumC +1, 0)               #Turning the empty position into 0 for visual checking 
            '''
            print('Round =',Movement,'\n',Bay)
            '''
            np.place(Bay, Bay == 0, NumC + 1)              #Turning the empty position into NumC + 1 to keep running the program 
            break                                          #After doing a reshuffle, stop the while loop
            '''
            print('\n')
            '''
        it = it + 100                                      #For stop the while loop, it doesn't matter in the program because the 'break' has stopped this while loop
    np.place(Bay, Bay == NumC +1, 0)
    '''
    print(Bay)
    '''
    '''
    np.place(Bay, Bay == 0, NumC + 1)
    '''
    '''
    print("Total movements =", Movement)
    print("Total relocation =", relocation)
    '''
    #print('Height =', Height)
    #print('Bay =', Bay)
    return Height                                          #Returning the column height after reshuffling

Bay_height = []                                            #Setting a list to record the column height of bay configuration before reshuffling
# =============================================================================
# The for loop records the column height before the bay configurations are reshuffled once
# =============================================================================
for i in range(0, len(Bay)):                               #For each bay configuration
    Height_origin = np.zeros((NCol, ), dtype = int)        #Creating an array to record column height and initializing the array as 0
    # =============================================================================
    # The nested for loop records column height of bay configuration  
    # =============================================================================
    for row in range(0, NTier):                            #Row              
        for column in range(0, NCol):                      #Column
            if Bay[i][row][column] > 0:                    #It means if there is any container in this postion
                Height_origin[column] += 1                 #The height of column plus 1
    Bay_height.append(Height_origin)                       #Recording the column height of the bay configuration

Bay_one_move = []                                          #Setting a list to record the column height of bay configuration after reshuffling 
# =============================================================================
# The for loop records the column height before the bay configurations are reshuffled once and apply the Min-Max heuristic to update the column height
# =============================================================================
for i in range(0, len(Bay)):                               #For each bay configuration
    Height = np.zeros((NCol, ), dtype = int)               #Creating an array to record column height and initializing the array as 0
    #print(Height)                                         
    #print(Bay[i])
    # =============================================================================
    # The nested for loop records column height of bay configuration  
    # =============================================================================
    for row in range(0, NTier):                            #Row
        for column in range(0, NCol):                      #Column
            if Bay[i][row][column] > 0 and Bay[i][row][column] < NumCon + 1: #It means if there is any container in this postion
                Height[column] += 1                                          #The height of column plus 1
    #print(Bay[i])
    Bay_test = bay_test(Bay[i], NumCon)                    #Turning 0 (empty position) into NumCon + 1 for applying the Min-Max heurstic
    Height = Min_Max(NTier, NCol, NumCon, Bay_test, Height)#Applying Min-Max heuristic to do reshuffle once and record the column height
    Bay_one_move.append(Height)                            #Recording the column height of the bay configuration

Bay_height = np.asarray(Bay_height)                        #Turning list into array
Bay_one_move = np.asarray(Bay_one_move)                    #Turning list into array

output_layer =  Bay_one_move - Bay_height                  #Calculating the difference of column height between bay configuration before reshuffle and after reshuffle

print(output_layer)


#create (12,m) output layer, but 2 one, others are zero
output_layer_bi = np.zeros((len(output_layer), NCol * 2), dtype = int)   #Creating a 2d-array to record output data
# =============================================================================
# The for loop records turn the difference of column height in each bay configuration into output data
# =============================================================================
for i in range(0, len(output_layer)):                                    #for column height of each bay configuration
    for j in range(0, NCol):                                             #for the height of each column
        if output_layer[i][j] == 1:                                      # 1 means the column that container is put on
            output_layer_bi[i][j + NCol] = 1                             #Recording the second half of output data which the container is put on  
            
        if output_layer[i][j] == -1:                                     # 1 means the column that container is taken 
            output_layer_bi[i][j] = 1                                    #Recording the first half of output data which the container is taken
      
print(output_layer_bi)                                               
print(output_layer_bi.shape)
output_layer_bi = np.transpose(output_layer_bi)                          #Doing transpose on the output data to apply ANN model
#print(output_layer_bi)


with open('output_layer_4_6_3_2.pickle' ,'wb') as file:                  ##Saving the generated output data as a pickle file
    pickle.dump(output_layer_bi, file)