# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 21:15:35 2018

@author: user
"""

import numpy as np
import pickle
import time
import random
np.set_printoptions(threshold=np.inf)
start = time.time()

NTier = int(4)
NCol = int(6)
NumCon = int(13)


with open('input_layer_4_6_13_2.pickle', 'rb') as file:
    input_layer_4_6_13_2 = pickle.load(file)
    
with open('input_layer_4_6_13_3.pickle', 'rb') as file:
    input_layer_4_6_13_3 = pickle.load(file)
    
with open('input_layer_4_6_13_4.pickle', 'rb') as file:
    input_layer_4_6_13_4 = pickle.load(file)


input_layer_trans = np.concatenate((input_layer_4_6_13_2, input_layer_4_6_13_3, input_layer_4_6_13_4), axis = 0)

input_layer = np.transpose(input_layer_trans)
Bay_dataset = input_layer_trans
Bay_dataset.shape = (len(Bay_dataset), NTier, NCol)

def bay_test(Initial_Bay):
    Bay_test = Initial_Bay
    np.place(Initial_Bay, Initial_Bay == 0, [NumCon + 1]) # strange syntax, transfer 0 to NumCon + 1
    return Bay_test

def height(Bay):
    height = np.zeros((NCol,), dtype = int)
    for i in range(0, NTier):
        for j in range(0, NCol):
            if Bay[i][j] >= 1:
                height[j] += 1
    return height


def Min_Max(nt,nc,NumC,Bay):
    it = 0
    Movement = 0
    relocation = 0    
    while it < NumC:        
        p_l_c = np.where(Bay == Bay.min(keepdims = True))  #p_l_c = The position of lowest container
        p_r = int(p_l_c[0]) #the row position of p_l_c
        p_c = int(p_l_c[1]) #the column position of p_l_c
        
        if  p_r == nt - int(Height[p_c]): #if target container is on the top of a stack, directly retrieving it
            Bay[p_r][p_c] = NumC + 1
            Height[p_c] = Height[p_c] - 1
            Movement += 1
            np.place(Bay, Bay == NumC +1, 0)
            #print('Round =',Movement,'\n',Bay)
            np.place(Bay, Bay == 0, NumC + 1)
            #print('\n')
        elif p_r > nt - Height[p_c]:
            r = nt - Height[p_c] 
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
                relocation += 1
                Movement += 1
                np.place(Bay, Bay == NumC +1, 0)
                #print('Round =',Movement,'\n',Bay)
                np.place(Bay, Bay == 0, NumC + 1)
                #print('relocation =', relocation)
                #print('\n')                                
                r = r + 1
                
            Bay[p_r][p_c] = NumC + 1
            Height[p_c] = Height[p_c] - 1
            Movement += 1
            np.place(Bay, Bay == NumC +1, 0)
            #print('Round =',Movement,'\n',Bay)
            np.place(Bay, Bay == 0, NumC + 1)
            #print('\n')
        it = it + 1
    np.place(Bay, Bay == NumC +1, 0)
    #print(Bay)
    '''
    np.place(Bay, Bay == 0, NumC + 1)
    '''
    #print("Total movements =", Movement)
    #print("Total relocation =", relocation)
    return Movement

movement_dataset = []
rounds = 0
for i in range(0, len(Bay_dataset)):
    '''
    Height = np.zeros((NCol,), dtype = int)
    for j in range(0, NTier):
        for k in range(0, NCol):
            if Bay_dataset[i][j][k] >= 1:
                Height[k] += 1
    '''
    Height = height(Bay_dataset[i])
    Bay_test = bay_test(Bay_dataset[i])
    Movement = Min_Max(NTier, NCol, NumCon, Bay_test)
    movement_dataset.append(Movement)
    rounds += 1
    print(rounds)
    
#print(movement_dataset)
print(sum(movement_dataset))
print(sum(movement_dataset) / len(movement_dataset))

end = time.time()
print(end - start)    


with open('result_min_max_4_6_13.pickle','wb') as file:
    pickle.dump(movement_dataset, file)

#315000
#5811242
#18.448387301587303
#274.9520936012268
    
    