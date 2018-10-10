import numpy as np 

def logLoss(y, predictions):
	# if np.shape(y) != np.shape(predictions):
	# 	print('shape y and shape predictions not the same')
	n = np.shape(y)[0]
	ones = np.ones((n,1))

	J = -1.0/n * ( np.dot( np.transpose(y), np.log(predictions) ) + 
		np.dot( np.transpose(ones-y),np.log(ones-predictions) ))

	return(J)
