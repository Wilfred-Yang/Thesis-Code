# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:39:58 2018

@author: user
"""

import itertools
import numpy as np
import random
import pickle
np.set_printoptions(threshold=np.inf)


NTier = int(4)
NCol = int(6)
NumCon = int(5)


with open('input_layer_4_6_5_4.pickle', 'rb') as file:
    Bay = pickle.load(file)

Bay.shape = (len(Bay), NTier, NCol)

def bay_test(Initial_Bay,NumC):
    Bay_test = Initial_Bay
    np.place(Initial_Bay, Initial_Bay == 0, NumC + 1) # strange syntax, transfer 0 to NumCon + 1
    return Bay_test

def Min_Max(nt,nc,NumC,Bay,Height):
    it = 0
    Movement = 0
    relocation = 0    
    while it < NumC:        
        p_l_c = np.where(Bay == Bay.min(keepdims = True))  #p_l_c = The position of lowest container
        '''
        print(Bay)
        
        print(p_l_c)
        '''
        p_r = int(p_l_c[0]) #the row position of p_l_c
        #print('p_r =', p_r)
        p_c = int(p_l_c[1]) #the column position of p_l_c
        if  p_r == nt - int(Height[p_c]): #if target container is on the top of a stack, directly retrieving it
            Bay[p_r][p_c] = NumC + 1            
            Height[p_c] = Height[p_c] - 1 
            '''
            print('Height =', Height) 
            '''
            Movement += 1
            np.place(Bay, Bay == NumC +1, 0)
            '''
            print('Round =',Movement,'\n',Bay)
            '''
            np.place(Bay, Bay == 0, NumC + 1)
            '''
            print('\n')
            '''
        elif p_r > nt - Height[p_c]:
            r = nt - Height[p_c]
            '''
            print('r = ', r)
            '''
            while p_r > r:   #while loop concept 
                i = 0
                Height_m = [] #create an empty list for column which is up to maximum height
                c_s_i = Bay.min(axis = 0) - Bay[r][p_c]  #candidate stack including target container
                while nc > i:
                    if Height[i] == nt:
                        Height_m.append(i)    #add column with maximum height to list 
                    i = i + 1
                Height_m.append(p_c)          # add the target column to list                   
                c_s = np.delete(c_s_i, Height_m, None) #candidate stack after deleting target container and stack up to height limit 
                
                if np.max(c_s) > 0:   #find arg number
                    arg_c = min(i for i in c_s if i > 0)
                else:
                    arg_c = max(c_s)
                    
                l_arg_c = []
                for i in range(0,nc):    # I ingore if there are many arg numbers, it will do the same things. It will influence the value of Height.
                    if c_s_i[i] == arg_c: # find the location of arg_c
                        l_arg_c.append(i) # add them to list
                r_arg_c = random.choice(l_arg_c) #random choose one of them
                Bay[nt-Height[r_arg_c]-1][r_arg_c] = Bay[r][p_c] #relocation
                Bay[r][p_c] = NumC +1
                Height[p_c] = Height[p_c] - 1
                Height[r_arg_c] = Height[r_arg_c] + 1
                '''
                print('\n')
                print('Height =', Height)
                '''
                relocation += 1
                Movement += 1
                np.place(Bay, Bay == NumC +1, 0)
                '''
                print('Round =',Movement,'\n',Bay)
                '''
                np.place(Bay, Bay == 0, NumC + 1)
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
            np.place(Bay, Bay == NumC +1, 0)
            '''
            print('Round =',Movement,'\n',Bay)
            '''
            np.place(Bay, Bay == 0, NumC + 1)
            break
            '''
            print('\n')
            '''
        it = it + 100
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
    return Height

Bay_height = []
for i in range(0, len(Bay)):
    Height_origin = np.zeros((NCol, ), dtype = int)
    for row in range(0, NTier):
        for column in range(0, NCol):
            if Bay[i][row][column] > 0:
                Height_origin[column] += 1    
    Bay_height.append(Height_origin)

Bay_one_move = []
for i in range(0, len(Bay)):
    Height = np.zeros((NCol, ), dtype = int)
    #print(Height)
    #print(Bay[i])
    for row in range(0, NTier):
        for column in range(0, NCol):
            if Bay[i][row][column] > 0 and Bay[i][row][column] < NumCon + 1:
                Height[column] += 1
    #print(Bay[i])
    Bay_test = bay_test(Bay[i], NumCon)
    Height = Min_Max(NTier, NCol, NumCon, Bay_test, Height)
    Bay_one_move.append(Height)

Bay_height = np.asarray(Bay_height)
Bay_one_move = np.asarray(Bay_one_move)

output_layer =  Bay_one_move - Bay_height

#create (12,m) output layer, but 2 one, others are zero
output_layer_bi = np.zeros((len(output_layer), NCol * 2), dtype = int)
for i in range(0, len(output_layer)):
    for j in range(0, NCol):
        if output_layer[i][j] == 1:
            output_layer_bi[i][j + NCol] = 1
            
        if output_layer[i][j] == -1:
            output_layer_bi[i][j] = 1

'''
output_layer_bi = np.zeros((len(output_layer), NCol), dtype = int) #(6, m)
for i in range(0, len(output_layer_bi)):
    for j in range(0, NCol):
        if output_layer[i][j] == 1:
            output_layer_bi[i][j] = 1
'''
#print(output_layer_bi)
output_layer_bi = np.transpose(output_layer_bi)
#print(output_layer_bi)
print(output_layer_bi.shape)


with open('output_layer_4_6_5_4.pickle' ,'wb') as file:
    pickle.dump(output_layer_bi, file)