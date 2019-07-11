# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:51:23 2019

@author: user
"""
# =============================================================================
# The program checks the performance of ANN-based system on each bay configuration. 
# The program compares bay configuration reshuffled by Min-Max heuristic with bay configuration reshuffled by ANN-based system (learning Min-Max heuristic)
# The program compares Look-ahead N and Better-of-Two with ANN-based system as well 
# =============================================================================
import numpy as np
import pickle

# =============================================================================
# Loading the movement of each bay configutation   
# =============================================================================
with open ('result_min_max.pickle','rb') as file:     #Loading the movement of each bay configuration reshuffled by Min-Max heuristic
    MM = pickle.load(file)

with open('LA_N_movement.pickle', 'rb') as file:      #Loading the movement of each bay configuration reshuffled by Look-ahead N heuristic
    LA = pickle.load(file)

with open('result_ML_min_max.pickle', 'rb') as file:  #Loading the movement of each bay configuration reshuffled by ANN-based system (learning Min-Max heuristic)
    ML_MM = pickle.load(file)

with open('LA_N_ML.pickle', 'rb') as file:            #Loading the movement of each bay configuration reshuffled by ANN-based system (learning Look-ahead N heuristic)
    ML_LA = pickle.load(file)

with open('combined_ML.pickle', 'rb') as file:        #Loading the movement of each bay configuration reshuffled by ANN-based system (learning Better-of-Two)
    ML_better_of_two = pickle.load(file)

with open('Better_of_two.pickle', 'rb') as file:      #Loading the movement of each bay configuration reshuffled by Better-of-Two
    Better_of_two = pickle.load(file)
    
print(len(MM))

# =============================================================================
# Turn list into numpy 
# =============================================================================
MM = np.asarray(MM)
LA = np.asarray(LA)
Better_of_two = np.asarray(Better_of_two)
ML_MM = np.asarray(ML_MM)
ML_LA = np.asarray(ML_LA)
ML_better_of_two = np.asarray(ML_better_of_two)

#print(sum(ML_MM) / (len(ML_MM))) 

# =============================================================================
# Calculating the different movement on each bay configuration 
# =============================================================================
Compare_LA = ML_LA - LA 
Compare_MM = ML_MM - MM
Compare_BOT = ML_better_of_two - Better_of_two


Worse_ML_LA = 0                   #Recording the number of bay configuration that ANN-based system (learning Look-ahead N heuristic) is worse than Look-ahead N
Better_ML_LA = 0                  #Recording the number of bay configuration that ANN-based system (learning Look-ahead N heuristic) is better than Look-ahead N
Worse_average_LA = []             #Recording the bay configuration that ANN-based system (learning Look-ahead N heuristic) is worse than Look-ahead N
Better_average_LA = []            #Recording the bay configuration that ANN-based system (learning Look-ahead N heuristic)is better than Look-ahead N

Worse_ML_MM = 0                   #Recording the number of bay configuration that ANN-based system (learning Min-Max heuristic) is worse than Look-ahead N
Better_ML_MM = 0                  #Recording the number of bay configuration that ANN-based system (learning Min-Max heuristic) is better than Look-ahead N
Worse_average_MM = []             #Recording the bay configuration that ANN-based system (learning Min-Max heuristic) is worse than Look-ahead N
Better_average_MM = []            #Recording the bay configuration that ANN-based system (learning Min-Max heuristic)is better than Look-ahead N


Worse_ML_BOT = 0                  #Recording the number of bay configuration that ANN-based system (learning Better-of-Two) is worse than Look-ahead N
Better_ML_BOT = 0                 #Recording the number of bay configuration that ANN-based system (learning Better-of-Two) is better than Look-ahead N
Worse_average_BOT = []            #Recording the bay configuration that ANN-based system (learning Better-of-Two) is worse than Look-ahead N
Better_average_BOT = []           #Recording the bay configuration that ANN-based system (learning Better-of-Two)is better than Look-ahead N

# =============================================================================
# The for loop records which method(heuristic or ANN-based system perform better in each bay configuration
# =============================================================================
for i in range(0, len(LA)):
    if Compare_LA[i] > 0: # ML is worse than the heuristic
        Worse_ML_LA += 1
        Worse_average_LA.append(Compare_LA[i])
        
    elif Compare_LA[i] < 0: #ML is better than the heuristic
        Better_ML_LA += 1
        Better_average_LA.append(Compare_LA[i])
        
    if Compare_MM[i] > 0:
        Worse_ML_MM += 1
        Worse_average_MM.append(Compare_MM[i])
        
    elif Compare_MM[i] < 0:
        Better_ML_MM += 1
        Better_average_MM.append(Compare_MM[i])
        
    if Compare_BOT[i] > 0:
        Worse_ML_BOT += 1
        Worse_average_BOT.append(Compare_BOT[i])
        
    elif Compare_BOT[i] < 0:
        Better_ML_BOT += 1
        Better_average_BOT.append(Compare_BOT[i])
        
print('Worse_ML_LA =', Worse_ML_LA)   
print('Better_ML_LA =', Better_ML_LA)
print('Worse_ML_MM =', Worse_ML_MM)
print('Better_ML_MM =', Better_ML_MM)
print('Worse_ML_BOT =', Worse_ML_BOT)
print('Better_ML_BOT =', Better_ML_BOT)

print('Worse_average_LA =', sum(Worse_average_LA)/ len(Worse_average_LA))
print('Better_average_LA =', sum(Better_average_LA)/ len(Better_average_LA))
print('Worse_average_MM =', sum(Worse_average_MM)/ len(Worse_average_MM))
print('Better_average_MM =', sum(Better_average_MM)/ len(Better_average_MM))
print('Worse_average_BOT =', sum(Worse_average_BOT)/ len(Worse_average_BOT))
print('Better_average_BOT =', sum(Better_average_BOT)/ len(Better_average_BOT))

print('Worse_average_LA =', sum(Worse_average_LA)/ len(Worse_average_LA))
print('Better_average_LA =', sum(Better_average_LA)/ len(Better_average_LA))
print('Worse_average_MM =', sum(Worse_average_MM)/ len(Worse_average_MM))
print('Better_average_MM =', sum(Better_average_MM)/ len(Better_average_MM))
print('Worse_average_BOT =', sum(Worse_average_BOT)/ len(Worse_average_BOT))
print('Better_average_BOT =', sum(Better_average_BOT)/ len(Better_average_BOT))
#Worse_ML_LA = 241610
#Better_ML_LA = 12510
#Worse_ML_MM = 113572
#Better_ML_MM = 4515
#Worse_ML_BOT = 238785
#Better_ML_BOT = 5517

#Worse_average_LA = 1.2371259467737263
#Better_average_LA = -1.0774580335731414
#Worse_average_MM = 1.1011428873313844
#Better_average_MM = -1.1140642303433002
#Worse_average_BOT = 1.2442699499549805
#Better_average_BOT = -1.044589450788472












