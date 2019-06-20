# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 21:46:56 2018

@author: user
"""

import numpy as np
import pickle
NTier = int(4)
NCol = int(6)
NumCon = int(15)
N = int(2)

with open('input_layer_4_6_15_4.pickle', 'rb') as file:
    input_layer = pickle.load(file)
#print(input_layer.shape)
input_layer_trans = input_layer

input_layer_trans.shape = (len(input_layer_trans), NTier, NCol)
#print(input_layer_trans)

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
    #print(n)
    #print(Bay)
    #print(Position)
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

#Height = Bay_height(Initial_Bay, NumCon)      
#print(Height)         
#Bay_test = bay_test(Initial_Bay)       
def present_con(Bay):
    NumC = np.max(Bay)
    
    return NumC


def LA_N(Bay_test, N, NumC):
    
    lowest_con = 1
    movement = 0
    relocation = 0
    
    NumC = 5 
    while lowest_con <= NumCon:       
        row_lowest_con, column_lowest_con = Find_con_position(Bay_test, lowest_con)   
        #print('row_lowest_con =', row_lowest_con, 'column_lowest_con =', column_lowest_con)
        r = 1
        N = min(2, NumC)
        #print('N =', N)
        #print('NumC =', NumC)
        
        while Bay_test[row_lowest_con - 1][column_lowest_con] <= NumCon and row_lowest_con != 0:            
            Stack_N = Find_Stack_N(Bay_test, N, lowest_con)
            Stack_Height, Stack_not_N = not_in_Stack_N(Stack_N, Height)
            #print(int(min(Stack_Height)))
            
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
            #np.place(Bay_test, Bay_test == 0, NumCon + 1)
            movement += 1
            relocation += 1
            break
        
        if movement == 0:
            
            Bay_test[row_lowest_con][column_lowest_con] = NumCon + 1
            Height[column_lowest_con] -= 1
            np.place(Bay_test, Bay_test == NumCon +1, 0)
        break

        #np.place(Bay_test, Bay_test == NumCon +1, 0)
        #print(Bay_test)
        np.place(Bay_test, Bay_test == 0, NumCon + 1)
        movement += 1
        lowest_con += 1
        NumC -= 1
        
    return Bay_test
                   
Height_list = []
Height_one_move = []

for i in range(0, len(input_layer_trans)):
    Height = Bay_height(input_layer_trans[i], NumCon)
    Height_list.append(Height)
Height_array = np.asarray(Height_list)


for i in range(0, len(input_layer_trans)):
    #print(input_layer_trans[i])
    Height = Bay_height(input_layer_trans[i], NumCon)
    NumC = present_con(input_layer_trans[i])
    #print('NumC =', NumC)
    Bay_test = bay_test(input_layer_trans[i])
    #print('Bay_test =', Bay_test)
    Calculation = LA_N(Bay_test, N, NumC)
    #print(Calculation)    
    Height_a_move = Bay_height(Calculation, NumCon)
    Height_one_move.append(Height_a_move)
    #print('\n')
    #print(Calculation)
    #print('\n')
    
  
Height_one_move = np.asarray(Height_one_move)
#print(Height_one_move)

output_layer = Height_one_move - Height_array 

#(12, m) output layer two 1 
output_layer_bi = np.zeros((len(input_layer_trans), NCol * 2), dtype = int)
for i in range(0, len(output_layer)):
    for j in range(0, NCol):
        if output_layer[i][j] == 1:
            output_layer_bi[i][j + NCol] = 1
            
        if output_layer[i][j] == -1:
            output_layer_bi[i][j] = 1
'''
 #(6, m)
output_layer_bi = np.zeros((len(output_layer), NCol), dtype = int) #(6,m)
for i in range(0, len(output_layer_bi)):
    for j in range(0, NCol):
        if output_layer[i][j] == 1:
            output_layer_bi[i][j] = 1
'''

#print(output_layer_bi)
print(output_layer_bi.shape)
output_layer_bi = np.transpose(output_layer_bi)
#print(output_layer_bi)
with open('output_layer_4_6_15_4.pickle', 'wb') as file:
    pickle.dump(output_layer_bi, file)

