# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:51:34 2018

@author: user
"""


import numpy as np
import pickle
import time
np.set_printoptions(threshold=np.inf)
start = time.time()

NTier = int(4)
NCol = int(6)
NumCon = int(14)
N = int(2)

with open('input_layer_4_6_14_2.pickle', 'rb') as file:
    input_layer_4_6_14_2 = pickle.load(file)
    
with open('input_layer_4_6_14_3.pickle', 'rb') as file:
    input_layer_4_6_14_3 = pickle.load(file)
    
with open('input_layer_4_6_14_4.pickle', 'rb') as file:
    input_layer_4_6_14_4 = pickle.load(file)


input_layer_trans = np.concatenate((input_layer_4_6_14_2, input_layer_4_6_14_3, input_layer_4_6_14_4), axis = 0)
#input_layer_trans = input_layer_trans[3300:3305, :]

input_layer = np.transpose(input_layer_trans)
Bay_dataset = input_layer_trans
Bay_dataset.shape = (len(Bay_dataset), NTier, NCol)

def bay_test(Initial_Bay):
    Bay_test = Initial_Bay
    np.place(Initial_Bay, Initial_Bay == 0, [NumCon + 1]) # strange syntax, transfer 0 to NumCon + 1
    return Bay_test

def Bay_height(Bay, NumCon):
    
    Height = np.zeros((NCol,), dtype = int)
    for i in range(0, NTier):
        for j in range(0, NCol):
            if Bay[i][j] > 0 and Bay[i][j] < NumCon + 1:
                Height[j] += 1
    return Height


def Find_Stack_N(Bay, N, lowest_con):
    Stack_N = []
    
    for i in range(lowest_con, lowest_con + N): # i equals tocontainer
        
        con_position = np.where(Bay == i)
        Stack_N.append(int(con_position[1]))
    
    return list(set(Stack_N))
           
def not_in_Stack_N(Stack_N, Height):
    
    Stack = []
    Stack_Height = []
    
    for i in range(0, NCol):
        Stack.append(i)
    
    for i in Stack_N:
        Stack.remove(i)
        
    for i in Stack:
        Stack_Height.append(Height[i])
            
    return Stack_Height, Stack # Stack = not in stack_N

def Find_Top_Stack_N(Bay, Stack_N):
    
    Top_Stack_N = []
    for i in Stack_N:
        for j in range(0, NTier):
            if Bay[j][i] < NumCon + 1 and Bay[j][i] != 0:
                Top_Stack_N.append(Bay[j][i])
                break
    
    return Top_Stack_N
                
def Find_con_position(Bay, n):
    
    Position = np.where(Bay == n)
    row = int(Position[0])
    column = int(Position[1])
    #print(row, column)
    return row, column

def Find_Lowest_con_in_column(Bay, Stack):  # Stack = not in stack_N
    Low_Stack = []
    #print(Stack)
    for i in Stack:
        con = NumCon + 1
        for j in range(0, NTier):
            if Bay[j][i] < con:
                con = Bay[j][i]
                #print(con)

        if Bay[0][i] == NumCon + 1:
            Low_Stack.append(con)

            
    
    return Low_Stack

def present_con(Bay):
    
    NumC = np.max(Bay)    
    return NumC


def LA_N(Bay_test, N):
    
    lowest_con = 1
    movement = 0
    relocation = 0
    NumC = 14
    while lowest_con <= NumCon:       
        row_lowest_con, column_lowest_con = Find_con_position(Bay_test, lowest_con)   
        #print('row_lowest_con =',row_lowest_con, 'column_lowest_con =', column_lowest_con)

        N = min(2, NumC)
        #print('N =', N)
        #print('NumC =', NumC)
        #print('lowest_con =', lowest_con)
        
        while Bay_test[row_lowest_con - 1][column_lowest_con] <= NumCon and row_lowest_con != 0:            
            Stack_N = Find_Stack_N(Bay_test, N, lowest_con)
            Stack_Height, Stack_not_N = not_in_Stack_N(Stack_N, Height)
            #print(int(min(Stack_Height)))
            r = 1
            while int(min(Stack_Height)) == NTier  or len(Stack_N) == NTier:              
                N -= 1
                Stack_N = Find_Stack_N(Bay_test, N, lowest_con)
                Stack_Height, Stack_not_N = not_in_Stack_N(Stack_N, Height) 
            #print('N =', N)
            #print('lowest container=', lowest_con)
            #print('Stack_N =', Stack_N)
            #print('Stack_Height =', Stack_Height)
            #print('Stack_not_N =', Stack_not_N)
            Top_Stack_N = Find_Top_Stack_N(Bay_test, Stack_N)
            #print('Top_Stack_N =', Top_Stack_N)
            Low_Stack = Find_Lowest_con_in_column(Bay_test, Stack_not_N)
            #print('Low_Stack =', Low_Stack)
            #print('r =', r)
            #print('len(Stack_N)  =', len(Stack_N))
            while r <= len(Stack_N):
                n = max(Top_Stack_N)
                #print('n = ',n)
                #print(lowest_con)
                row_n, column_n = Find_con_position(Bay_test, n)
                #print('row_n =', row_n, 'column_n =', column_n)
                
                if column_n == column_lowest_con:
                    if min(Low_Stack) == NumCon + 1:
                        for j in range(0, NCol):                           
                            if Bay_test[NTier-1][j] == NumCon + 1:
                                row_low, column_low = NTier, j
                                #print('row_low = ',row_low, 'column_low =', column_low)
                                break
                            
                    elif max(Low_Stack) - n > 0:
                        while min(Low_Stack) - n < 0:
                            Low_Stack.remove(min(Low_Stack))
                            
                        if min(Low_Stack) == NumCon + 1:
                            for j in range(0, NCol):                            
                                if Bay_test[NTier-1][j] == NumCon + 1:
                                    row_low, column_low = NTier, j
                                    #print('row_low = ',row_low, 'column_low =', column_low)
                        else:                          
                            row, column_low = Find_con_position(Bay_test, min(Low_Stack))
                            row_low = NTier - Height[column_low] 
                        
                    else:                     
                        row, column_low = Find_con_position(Bay_test, max(Low_Stack))
                        row_low = NTier - Height[column_low] 
                        #print('row_low =', row_low, 'column_low =', column_low)
                        
                    break
                
                elif Bay_test[row_n][column_n] <= min(list(Bay_test[i][column_n] for i in range(0, NTier))):
                    #print('min =', min(list(Bay_test[i][column_n] for i in range(0, NTier))))
                    for i in Stack_N:
                        rounds = 0
                        for j in range(0, NTier-1):
                            
                            if Bay_test[j][i] < Bay_test[j+1][i]:
                                rounds += 1
                               
                            elif Bay_test[j][i] == NumCon + 1:
                                rounds += 1
                                
                        if Bay_test[i][NTier] != NumCon + 1:
                            rounds += 1
                        
                        if rounds == NTier:
                            Stack_N.remove(i)
                    
                               
                    if r == len(Stack_N):
                        row, column_low = Find_con_position(Bay_test, min(Low_Stack))   
                        row_low = NTier - Height[column_low] 
                        break
                    
                    else:
                        r += 1                    
                                
                elif max(Low_Stack) > n:
                                        
                    if min(Low_Stack) == NumCon + 1:
                         
                        for j in range(0, NCol):
                            
                            if Bay_test[NTier-1][j] == NumCon + 1:
                                row_low, column_low = NTier, j
                                #print('row_low = ',row_low, 'column_low =', column_low)
                                break
                    else:
                        while min(Low_Stack) - n < 0:
                           Low_Stack.remove(min(Low_Stack))
                           
                        if min(Low_Stack) == NumCon + 1:
                            for j in range(0, NCol):
                                if Bay_test[NTier-1][j] == NumCon + 1:
                                    row_low, column_low = NTier, j
                                    #print('row_low = ',row_low, 'column_low =', column_low)
                                    break
                        else:
                            row, column_low = Find_con_position(Bay_test, min(Low_Stack))
                            row_low = NTier - Height[column_low] 
                                
                    #print(min(Low_Stack))                  
                    break
                
                elif r == len(Stack_N):
                    
                    row, column_low = Find_con_position(Bay_test, max(Low_Stack))
                    row_low = NTier - Height[column_low] 
                    
                    break
                
                Top_Stack_N.remove(max(Top_Stack_N))
                #print('Top_Stack_N =', Top_Stack_N)
                r += 1

            Bay_test[row_low-1][column_low] = Bay_test[row_n][column_n]
            Bay_test[row_n][column_n] = NumCon + 1
            Height[column_low] += 1
            Height[column_n] -= 1

            np.place(Bay_test, Bay_test == NumCon +1, 0)
            
            #print(Bay_test)
            np.place(Bay_test, Bay_test == 0, NumCon + 1)
            movement += 1
            relocation += 1
            
        Bay_test[row_lowest_con][column_lowest_con] = NumCon + 1
        Height[column_lowest_con] -= 1
        np.place(Bay_test, Bay_test == NumCon +1, 0)
        #print(Bay_test)
        np.place(Bay_test, Bay_test == 0, NumCon + 1)
        movement += 1
        lowest_con += 1
        NumC -= 1
        
    return movement

rounds = 0
movement_dataset = []
#print(Bay_dataset[62793])

for i in range(0, len(Bay_dataset)):
    Height = np.zeros((NCol,), dtype = int)
    for j in range(0, NTier):
        for k in range(0, NCol):
            if Bay_dataset[i][j][k] >= 1:
                Height[k] += 1

    #Height = height(Bay_dataset[i])
    #print(Bay_dataset[i])
    Bay_test = bay_test(Bay_dataset[i])
    #print(Bay_test)
    Movement = LA_N(Bay_test, N)
    #print(Movement)
    movement_dataset.append(Movement)
    rounds += 1
    #print(rounds)
    print(len(movement_dataset))

with open('LA_N_movement_4_6_14.pickle', 'wb') as file:
    pickle.dump(movement_dataset, file)
   
print(sum(movement_dataset))
print(sum(movement_dataset) / len(movement_dataset))


end = time.time()
print(end - start) 

#6336910
#20.117174603174604
#230.74934911727905
