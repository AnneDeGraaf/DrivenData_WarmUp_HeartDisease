import numpy as np 

def logLoss(y, predictions):
	n = np.shape(y)[0]
	ones = np.ones((n,1))

	J = -1.0/n * ( np.dot( np.transpose(y), np.log(predictions) ) + 
		np.dot( np.transpose(ones-y),np.log(ones-predictions) ))

	return(J)



def logLoss2(y,predictions):
	if len(y) != len(predictions):
		print('length y and length predictions not the same')
		print(len(y), len(predictions))
		return(1)

	n = len(y)
	J = 0
	for i in range(n):
		J += -1.0/n * (y[i] * np.log(predictions[i]) + 
			(1.0-y[i])*np.log(1.0-predictions[i]))
	return(J)
