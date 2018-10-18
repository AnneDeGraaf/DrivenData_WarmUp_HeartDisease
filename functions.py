import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression




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


def crossVal_kFold(X, y):
	LR_model = LogisticRegression(penalty='l2', tol=1e-8, solver='liblinear')
	k = 10
	k_fold = StratifiedKFold(k, shuffle=False)
	k_score = np.zeros(k)
	f_score = np.zeros(np.shape(X)[1])
	j = 0
	for f in range(np.shape(X)[1]):
		feature = np.transpose(([X[:,f]]))
		i = 0
		for iTrain, iCross in k_fold.split(feature, y):
		    X_train, X_cross = feature[iTrain], feature[iCross]
		    y_train, y_cross = y[iTrain], y[iCross]
		    model_i = LR_model.fit(X_train, np.ravel(y_train))
		    predictions_i = model_i.predict_proba(X_cross)
		    k_score[i] = logLoss(y_cross, predictions_i)[0,1]
		    i += 1
		f_score[j] = 1.0/np.mean(k_score)  # SelectKBest selects on highest score, so need to invert
		j += 1
	return(f_score)

## THIS FUNCTION DOES NOT MAKE ANY SENSE
# def bestFeatures(X,y):
# 	LR_model = LogisticRegression(penalty='l2', tol=1e-8, solver='liblinear')
# 	f_score = np.zeros(np.shape(X)[1])
# 	i = 0
# 	for f in range(np.shape(X)[1]):
# 		feature = np.transpose(([X[:,f]]))
# 		model_i = LR_model.fit(feature, np.ravel(y))
# 		predictions_i = model_i.predict_proba(feature)
# 		k_score_i = logLoss(y, predictions_i)[0,1]
# 		f_score[i] = 1.0/k_score_i
# 		i += 1
# 	return(f_score)


















