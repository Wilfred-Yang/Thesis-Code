# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 18:55:02 2018

@author: user
"""

# =============================================================================
# The program apply the Look-ahead N heuristic to create output data that applies to ANN model.
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
#        (3) Applying the Look-ahead N heuristic to do reshuffle once and update the column height of bay configuration
#        
#     2. Calculate the difference between the column height before reshuffling and column height after reshuffling
#     3. Turning the difference in step 2 into output data
# =============================================================================

# =============================================================================
# The concept of the Min-Max heuristic won't be explained here. 
# If you are highly interested in the Min-Max heuristic, free feel to reference the study 
# "A new mixed integer program and extended look-ahead heuristic algorithm for the block relocation problem"
# =============================================================================

import numpy as np
import pickle
# =============================================================================
#Deciding the size and container of the bay
# =============================================================================
NTier = int(4)  #Number of rows
NCol = int(6)   #Number of columns
NumCon = int(3) #Number of containers
N = int(2)      #N represents the lowest N number containers in the bay. If there are 5 containers (container 1,2,3,4,5), 
                #container 1 and 2 are lowest number containers 

with open('input_layer_4_6_3_2.pickle', 'rb') as file:           #Loading the file of input data 
    input_layer = pickle.load(file)
#print(input_layer.shape)
input_layer_trans = input_layer

input_layer_trans.shape = (len(input_layer_trans), NTier, NCol)  #Change the shape of input data to trun them into bay configurations
print(input_layer_trans)

# =============================================================================
# The function is for turning 0 (empty space) into higher number (Numc + 1) that enables the Min-Max heuristic to do reshuffle in Look-ahead N heuristic
# =============================================================================
def bay_test(Initial_Bay):
    Bay_test = Initial_Bay
    np.place(Initial_Bay, Initial_Bay == 0, [NumCon + 1])        #transfer 0 (empty position) to NumCon + 1
    return Bay_test

# =============================================================================
# The function calculates the column height of bay 
# =============================================================================
def Bay_height(Bay, NumCon):                                     #(Bay configuration, number of containers)
    
    Height = np.zeros((NCol,), dtype = int)                      #Creating an array to record column height and initializing the array as 0
    for i in range(0, NTier):                                    #Row
        for j in range(0, NCol):                                 #Column
            if Bay[i][j] > 0 and Bay[i][j] < NumCon + 1:         #It means if there is any container in this postion
                Height[j] += 1                                   #The height of column j plus 1
    return Height                                                #Returning the column height of the bay configuration

# =============================================================================
# The function finds columns where the lowest N containers exist
# =============================================================================
def Find_Stack_N(Bay, N, lowest_con):                            #(Bay configuration, N, label of lowest number container)
    Stack_N = []                                                 #Creating a list for recording columns where the lowest N containers exist
    
    for i in range(lowest_con, lowest_con + N):                  #For each container of N lowest number containers
        
        con_position = np.where(Bay == i)                        #Find the position of container i 
        Stack_N.append(int(con_position[1]))                     #Record the column of container i
    
    return list(set(Stack_N))                                    #Sort the list of column and return
# =============================================================================
# The function finds columns where the lowest N containers don't exist and records the column height    
# =============================================================================
def not_in_Stack_N(Stack_N, Height):                             # (the list from Find_Stack_N function, column height of the bay)
    
    Stack = []                                                   #Setting a list that records columns where the lowest N containers don't exist
    Stack_Height = []                                            #Setting a list that records column height of the list Stack
    
    for i in range(0, NCol):                                     #The for loop records all columns to the list Stack
        Stack.append(i)
    
    for i in Stack_N:                                            #Removing column where the lowest N containers exist out of the list Stack
        Stack.remove(i)
        
    for i in Stack:                                              #Recording the column height of the list Stack  
        Stack_Height.append(Height[i])
            
    return Stack_Height, Stack 

# =============================================================================
# The function finds container on the top of the column where the lowest N containers exist
# =============================================================================
def Find_Top_Stack_N(Bay, Stack_N):                              #(Bay, list that records the column where the lowest N containers exist)
    
    Top_Stack_N = []                                             #Setting a list to record container on the top of column where the lowest N containers exist
    for i in Stack_N:                                            #for column that involved in Stack_N
        for j in range(0, NTier):                                #Row
            if Bay[j][i] < NumCon + 1 and Bay[j][i] != 0:        #The if condition finds the container on the top of the column 
                Top_Stack_N.append(Bay[j][i])
                break                                      
    
    return Top_Stack_N

# =============================================================================
# The function finds the position (row, column) of container in the bay 
# =============================================================================
def Find_con_position(Bay, n):                                   #(Bay, the label (number) of container)
    
    Position = np.where(Bay == n)                                #Finding the postion of container n
    row = int(Position[0])                                       #Getting the row
    column = int(Position[1])                                    #Getting the column
    #print(row, column)
    return row, column
# =============================================================================
# The function finds the lowset number container in the column
# =============================================================================
def Find_Lowest_con_in_column(Bay, Stack):                       #(Bay, columns of list from the function not_in_Stack_N) 
    Low_Stack = []                                               #Setting a list to record the lowest of the column
    #print(Stack)

    for i in Stack:                                              #For each column where the lowest number containers don't exist 
        con = NumCon + 1
        for j in range(0, NTier):                                #Row
            if Bay[j][i] < con:                                  #The if condition compares container in the column to find the lowest number container
                con = Bay[j][i]
                #print(con)

        if Bay[0][i] == NumCon + 1:                              #The if condition ensures any empty position in the column so that container can be put on
            Low_Stack.append(con)

            
    
    return Low_Stack

#Height = Bay_height(Initial_Bay, NumCon)      
#print(Height)         
#Bay_test = bay_test(Initial_Bay)
    
# =============================================================================
# The function finds the highest number container in the bay       
# =============================================================================
def present_con(Bay):
    NumC = np.max(Bay)
    
    return NumC

# =============================================================================
# The function is Look-ahead heuristic
# =============================================================================
def LA_N(Bay_test, N, NumC): #(Bay, the number of N, the number of present container in the bay)
    
    lowest_con = 1           #Setting the lowest number container to be container 1
    movement = 0             #For calculating number of movements (reshuffles + retrievals) to empty the bay 
    relocation = 0           #For calculating number of reshuffles to empty the bay 
    
    #NumC = 7                #Ignore
    # =============================================================================
    # The while loop ensures that every container will be retrieved from the bay  
    # =============================================================================
    while lowest_con <= NumCon:                                                             #NumCon = Number of containers in the initial bay
        row_lowest_con, column_lowest_con = Find_con_position(Bay_test, lowest_con)         #Find the position of lowest number container
        r = 1
        N = min(2, NumC)                                                                    #Deciding the value of N, I have initialized it as 2 to be compared with NumC
        #print('N =', N)
        #print('NumC =', NumC)
        
        while Bay_test[row_lowest_con - 1][column_lowest_con] <= NumCon and row_lowest_con != 0: #The while loop ensures there is no container above the lowest number container       
            Stack_N = Find_Stack_N(Bay_test, N, lowest_con)                                      #Finding columns where the lowest N containers exist
            Stack_Height, Stack_not_N = not_in_Stack_N(Stack_N, Height)                          #Finding columns and the column height where the lowest N containers don't exist 
            #print(int(min(Stack_Height)))
            
            while int(min(Stack_Height)) == NTier  or len(Stack_N) == NCol:                     #The while loop ensures that columns are in Stack_N not full and Stack_N does't include all column of the bay 
                                                                                                #It makes container unable to do reshuffle
            #while int(min(Stack_Height)) == NTier  or len(Stack_N) == NTier:
                N -= 1                                                                          #Let N - 1 and execute the two functions again until container can be reshuffled
                Stack_N = Find_Stack_N(Bay_test, N, lowest_con)
                Stack_Height, Stack_not_N = not_in_Stack_N(Stack_N, Height) 
            #print('N =', N)
            #print('lowest container=', lowest_con)
            #print('Stack_N =', Stack_N)
            #print('Stack_Height =', Stack_Height)
            #print('Stack_not_N =', Stack_not_N)
            Top_Stack_N = Find_Top_Stack_N(Bay_test, Stack_N)                                   #Finding the top position of container in Stack_N to find containers that should be reshuffled in higher priority
            #print('Top_Stack_N =', Top_Stack_N)
            Low_Stack = Find_Lowest_con_in_column(Bay_test, Stack_not_N)                        #Finding the lowest number container for the columns that the lowest N container don't exist
                                                                                                #So that the program can compare containers to do reshuffle
            #print('Low_Stack =', Low_Stack)
            #print('r =', r)
            #print('len(Stack_N)  =', len(Stack_N))
            # =============================================================================
            # The while loop finds container to reshuffle and where the container is reshuffled             
            # =============================================================================
            while r <= len(Stack_N):                                                            #The while loop set order about the columns in Stack_N and start to do reshuffle
                                                                                                #If the reshuffle causes deadlock in the first column,  then stopping the reshuffle and trying to reshuffle container from the second column
                n = max(Top_Stack_N)                                                            #Find the highest number container (container n) in Top_Stack_N
                #print('n = ',n)
                #print(lowest_con)
                row_n, column_n = Find_con_position(Bay_test, n)                                #Find the position of container n
                #print('row_n =', row_n, 'column_n =', column_n)
                
                # =============================================================================
                # The if-elif condition set priorties in reshuffling container. In this if condition, 
                # reshuffling container n if container n is the column where target container exists (No matter causing deadlock or not)
                # =============================================================================
                if column_n == column_lowest_con:                                               #Checking if container n is in the column where target container exists
                    if min(Low_Stack) == NumCon + 1:                                            #If columns where the lowest N containers don't exist are all empty column
                        for j in range(0, NCol):                                                #The for loop record a position of empty space that container n is reshuffled to  
                            if Bay_test[NTier-1][j] == NumCon + 1:
                                row_low, column_low = NTier, j
                                #print('row_low = ',row_low, 'column_low =', column_low)
                                break                                                           #Stop the for loop because the program find a position to do reshuffle
                            
                    elif max(Low_Stack) - n > 0:                                                #The if condition checks whether the reshuffle causes a new deadlock or not
                        while min(Low_Stack) - n < 0:                                           #The while loop finds the best column for container n to do reshuffle
                            Low_Stack.remove(min(Low_Stack))
                                                    
                        # =============================================================================
                        #The if condition finds row and column for container n to do reshuffle
                        #The included part is unnecessnary, feel free to skip it. Bacause programs of generate output data have followed the sample, I don't revise it more now
                        if min(Low_Stack) == NumCon + 1:                                         
                            for j in range(0, NCol):                            
                                if Bay_test[NTier-1][j] == NumCon + 1:                                   
                                    row_low, column_low = NTier, j
                                    #print('row_low = ',row_low, 'column_low =', column_low)
                        else:                          
                        # =============================================================================
                            row, column_low = Find_con_position(Bay_test, min(Low_Stack))       #Find the row and column in the following two lines
                            row_low = NTier - Height[column_low]

                        # =============================================================================
                        # It should be like this after the uncessnary if condition
                        # =============================================================================
                        #row, column_low = Find_con_position(Bay_test, min(Low_Stack))       
                        #row_low = NTier - Height[column_low]

                        
                    else:                                                                      #If the reshuffle causes deadlock, finding the closest number of container between container n 
                                                                                               #and container in Low_Stack to ensure smooth reshuffles in the future
                        row, column_low = Find_con_position(Bay_test, max(Low_Stack))          #Find the row and column in the following two lines
                        row_low = NTier - Height[column_low] 
                        #print('row_low =', row_low, 'column_low =', column_low)
                        
                    break                                                                      #Stopping the while loop because row, column and container n is found
                
                # =============================================================================
                # The condition checks whehter container n is the lowest numeber in the column. If yes, container is included in the lowest N containers. 
                # The program will consider whether reshuffling container n or not, because the container n may can wait to be retrieved rather than immidiately doing reshuffle
                # =============================================================================
                elif Bay_test[row_n][column_n] <= min(list(Bay_test[i][column_n] for i in range(0, NTier))):
                    #print('min =', min(list(Bay_test[i][column_n] for i in range(0, NTier))))
                    
                    
                    # =============================================================================
                    # The program is uncessnary here, feel free to skip it

                    #for i in Stack_N:                                                          
                    #    rounds = 0
                    #    for j in range(0, NTier-1):
                            
                    #        if Bay_test[j][i] < Bay_test[j+1][i]:
                    #            rounds += 1
                               
                    #        elif Bay_test[j][i] == NumCon + 1:
                    #            rounds += 1
                                
                    #    if Bay_test[i][NTier] != NumCon + 1:
                    #        rounds += 1
                        
                    #    if rounds == NTier:
                    #        Stack_N.remove(i)
                    # =============================================================================
                               
                    if r == len(Stack_N):                                                    #If container n is the last in the Stack_N, the program have to reshffle 
                                                                                             #container n whether causing a new deadlock or not
                        row, column_low = Find_con_position(Bay_test, min(Low_Stack))        #Finding the row and column in the following two lines
                        row_low = NTier - Height[column_low] 
                        break                                                                #Stopping the while loop because row, column and container n is found
                    
                    else:                                                                    #If not, the program will change container in Top_Stack_n to do reshuffle
                        r += 1                    
                # =============================================================================
                # The condition deals with container when the reshuffle don't cause a new deadlock        
                # =============================================================================
                elif max(Low_Stack) > n:                                                     #If there is a container in Low_Stack higher than container n 
                                        
                    if min(Low_Stack) == NumCon + 1:                                         #The if condition ensures the columns that container n may be reshuffle are all empty 
                         
                        for j in range(0, NCol):                                             #The for loop record a position of empty space that container n is reshuffled to
                            
                            if Bay_test[NTier-1][j] == NumCon + 1:
                                row_low, column_low = NTier, j
                                #print('row_low = ',row_low, 'column_low =', column_low)
                                break
                    else:                                                                    #If those columns aren't all empty
                        while min(Low_Stack) - n < 0:                                        #Utilizing while loop to find the best column for container n to do reshuffle
                           Low_Stack.remove(min(Low_Stack)) 
                        row, column_low = Find_con_position(Bay_test, min(Low_Stack))        #Find the row and column in the following two lines
                        row_low = NTier - Height[column_low] 
                                
                    #print(min(Low_Stack))                  
                    break                                                                    #Stopping the while loop because row, column and container n is found
                    
                
                # =============================================================================
                # The condition ensures the last container in Top_Stack_N that needs to be reshuffled whether causing a new deadlock or not                
                # =============================================================================
                elif r == len(Stack_N):                                                      #Ensuring the last container in Top_Stack_N  
                    
                    row, column_low = Find_con_position(Bay_test, max(Low_Stack))            #Find the row and column in the following two lines
                    row_low = NTier - Height[column_low] 
                    
                    break                                                                    #Stopping the while loop because row, column and container n is found
                
                Top_Stack_N.remove(max(Top_Stack_N))                                         #If the container n doesn't match any conditions above, 
                                                                                             #the program won't reshuffle container n and turning to the next container
                #print('Top_Stack_N =', Top_Stack_N)
                r += 1                                                                      

            Bay_test[row_low-1][column_low] = Bay_test[row_n][column_n]                      #The following two lines reshuffle the container n
            Bay_test[row_n][column_n] = NumCon + 1
            Height[column_low] += 1                                                          #The following two lines change the column height after reshuffling container n
            Height[column_n] -= 1

            np.place(Bay_test, Bay_test == NumCon +1, 0)                                     #Turning the empty position into 0 for visual checking 
            
            #print(Bay_test)
            #np.place(Bay_test, Bay_test == 0, NumCon + 1)
            movement += 1
            relocation += 1
            break                                                                            #Stopping the while loop because the program only needs a reshuffle here
        
# =============================================================================
#         if movement == 0:                                                                  #Not necessary here
#             
#             Bay_test[row_lowest_con][column_lowest_con] = NumCon + 1
#             Height[column_lowest_con] -= 1
#             np.place(Bay_test, Bay_test == NumCon +1, 0)
# =============================================================================
        
        break                                                                                #Stopping the while loop because the program only needs a reshuffle here

        #np.place(Bay_test, Bay_test == NumCon +1, 0)
        #print(Bay_test)
        np.place(Bay_test, Bay_test == 0, NumCon + 1)                                        #The following 4 lines aren't necessary here   
        movement += 1
        lowest_con += 1
        NumC -= 1
        
    return Bay_test                                                                          #Returning the bay after doing a reshuffle
                   
Height_list = []                                                                             #Setting a list to record the column height of bay configurations before reshuffling
Height_one_move = []                                                                         #Setting a list to record the column height of bay configurations after reshuffling 

# =============================================================================
# The for loop helps to record the column height of each bay configuration before reshuffling
# =============================================================================
for i in range(0, len(input_layer_trans)):                                                
    Height = Bay_height(input_layer_trans[i], NumCon)                                        #Applying the function to get column height
    Height_list.append(Height)                                                               #Recording it to the list
Height_array = np.asarray(Height_list)                                                       #Turing the list into array

# =============================================================================
# The for loop applies Look-ahead N heuristic and records the column height of each bay configuration after reshuffling
# =============================================================================
for i in range(0, len(input_layer_trans)):                                                  
    #print(input_layer_trans[i])
    Height = Bay_height(input_layer_trans[i], NumCon)                                        #Applying the function to get column height before reshuffling
    NumC = present_con(input_layer_trans[i])                                                 #Getting the highest number of container in the bay 
    Bay_test = bay_test(input_layer_trans[i])                                                #Turning 0 (empty position) into NumCon + 1 for applying the Look-ahead N heurstic
    Calculation = LA_N(Bay_test, N, NumC)                                                    #Reshuffling once and get the bay configuration after reshuffling once
    #print(Calculation)    
    Height_a_move = Bay_height(Calculation, NumCon)                                          #Applying the function to get column height after reshuffling     
    Height_one_move.append(Height_a_move)                                                    #Recording it to the list
    #print('\n')
    #print(Calculation)
    #print('\n')
    
  
Height_one_move = np.asarray(Height_one_move)                                                #Turning list into array
#print(Height_one_move)

output_layer = Height_one_move - Height_array                                                #Calculating the difference of column height between bay configuration before reshuffle and after reshuffle

#(12, m) output layer two 1 
output_layer_bi = np.zeros((len(input_layer_trans), NCol * 2), dtype = int)                  #Creating a 2d-array to record output data
# =============================================================================
# The for loop records turn the difference of column height in each bay configuration into output data
# =============================================================================
for i in range(0, len(output_layer)):                                                        #for column height of each bay configuration
    for j in range(0, NCol):                                                                 #for the height of each column
        if output_layer[i][j] == 1:                                                          # 1 means the column that container is put on
            output_layer_bi[i][j + NCol] = 1                                                 #Recording the second half of output data which the container is put on  
            
        if output_layer[i][j] == -1:                                                         # 1 means the column that container is taken 
            output_layer_bi[i][j] = 1                                                        #Recording the first half of output data which the container is taken


print(output_layer_bi)

output_layer_bi = np.transpose(output_layer_bi)                                              #Doing transpose on the output data to apply ANN model
#print(output_layer_bi)
with open('output_layer_4_6_3_2.pickle', 'wb') as file:                                      #Saving the generated output data as a pickle file
    pickle.dump(output_layer_bi, file)