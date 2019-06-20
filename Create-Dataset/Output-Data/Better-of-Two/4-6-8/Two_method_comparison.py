# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:21:35 2018

@author: user
"""

import pickle
import numpy as np

with open('output_layer_4_6_8_2_MM.pickle', 'rb') as file:
    output_layer_4_6_8_2_MM = pickle.load(file)
    
with open('output_layer_4_6_8_3_MM.pickle', 'rb') as file:
    output_layer_4_6_8_3_MM = pickle.load(file)

with open('output_layer_4_6_8_4_MM.pickle', 'rb') as file:
    output_layer_4_6_8_4_MM = pickle.load(file)
    
with open('output_layer_4_6_8_2_LA.pickle', 'rb') as file:
    output_layer_4_6_8_2_LA = pickle.load(file)
    
with open('output_layer_4_6_8_3_LA.pickle', 'rb') as file:
    output_layer_4_6_8_3_LA = pickle.load(file)   

with open('output_layer_4_6_8_4_LA.pickle', 'rb') as file:
    output_layer_4_6_8_4_LA = pickle.load(file)
    
#print(output_layer_4_6_8_2_LA.shape)
#print(output_layer_4_6_8_3_LA.shape) 
#print(output_layer_4_6_8_4_LA.shape) 
MM = np.concatenate((output_layer_4_6_8_2_MM, output_layer_4_6_8_3_MM, output_layer_4_6_8_4_MM), axis = 1)
LA = np.concatenate((output_layer_4_6_8_2_LA, output_layer_4_6_8_3_LA, output_layer_4_6_8_4_LA), axis = 1)

MM = np.transpose(MM)
LA = np.transpose(LA)
#print(MM.shape)


with open('LA_N_movement_4_6_8.pickle','rb') as file:
    LA_N = pickle.load(file)
    
with open('result_min_max_4_6_8.pickle','rb') as file:
    Min_Max = pickle.load(file)
    
LA_N = np.asarray(LA_N)

Min_Max = np.asarray(Min_Max)

#print(Min_Max)

'''
for i in range(0, len(compare)):

    if compare[i] > 0:
        Min.append(i)

print(len(Min))
print(Min)

for i in range(0, len(compare_LA_N)):

    if compare_LA_N[i] > 0:
        LA.append(i)

print(len(LA))
print(LA)
'''


compare = Min_Max - LA_N

#print(compare)
MM_better_LA = []
LA_better_MM = []
for i in range(0, len(compare)):

    if compare[i] > 0:
        LA_better_MM.append(i)
    
    if compare[i] < 0:
        MM_better_LA.append(i)



#print(LA_better_MM)
#print(MM_better_LA)
print(len(LA_better_MM))
print(len(MM_better_LA))
#print(MM.shape)
#print(LA.shape)

for i in LA_better_MM:
    #print(i)
    #print(MM[i])
    MM[i] = LA[i]

print(MM.shape)


output_layer_4_6_8_2_combine = MM[0:205000, :]
output_layer_4_6_8_3_combine = MM[205000:5980840, :]
output_layer_4_6_8_4_combine = MM[5980840:6385840, :]


with open('output_layer_4_6_8_combine.pickle', 'wb') as file:
    pickle.dump(MM, file)

with open('output_layer_4_6_8_2_combine.pickle', 'wb') as file:
    pickle.dump(output_layer_4_6_8_2_combine, file)

with open('output_layer_4_6_8_3_combine.pickle', 'wb') as file:
    pickle.dump(output_layer_4_6_8_3_combine, file)
    
with open('output_layer_4_6_8_4_combine.pickle', 'wb') as file:
    pickle.dump(output_layer_4_6_8_4_combine, file)  


