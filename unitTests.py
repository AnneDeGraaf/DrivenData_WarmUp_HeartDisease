import numpy as np
from logLoss import logLoss

# Test logLoss function
y = np.array([[1], [1], [0], [1], [0]])
p = np.array([[0.5], [0.70], [0.9], [0.1], [0.1]])
ans_function = round(logLoss(y,p), 4)
ans_true = 1.1521
if ans_true==ans_function:
	print('logLoss function works!')
else:
	print('logLoss function output incorrect')
	print('output', ans_function, 'should be', ans_true)




