There are samples showing how dataset is generated. They are 'Input-data-sample1', 'Input-data-sample2', 'Output-data-LA-sample', 
'Output-data-MM-sample', and 'Output-data-BOA-sample'.

In generating input data, I treat present situation of container bay as input data. In my study, I use 4-row, 6-column, and 18-container
bay. The idea of generating a bay is creating a 2d-array and the size is (4, 6). By shaping the size from (4, 6) to (24, 1) that matches 
the size requirement of ANN. 

In the thesis, containers are with integer. Thus, there are many possibilities for containers put in different positions. Cartesian product 
is applied to generate all possible container bay when number of container is less than or equal to 7. The program is shown in 
'Input-data-sample1'. Because all combination of bay configuration will be too large for training the ANN when the number is greater than 7. For 
container >= 8, 'Input-data-sample2' randomly generate destined number of input-data without applying Cartesian product.

In generating output-data, the program turns a reshuffle into output-data. The stack pattern of reshuffle is the output-data. If column = 6, the size 
of a output-data is (12, 1). They are all binary. The first half of number shows the column that the container is taken. The second half of 
number shows the column that the container is put. 

Here is the connection between input-data and output-data. The input-data shows bay configurations. The output-data shows the 1 reshuffle of 
bay configurations from input-data. 

'Output-data-LA-sample', 'Output-data-MM-sample', and 'Output-data-BOA-sample' apply different heuristics to reshuffle container. They are 
Look-ahead N, Min-Max, and Better-of-Two.