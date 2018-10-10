from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np 
from logLoss import logLoss

# load train data
allFeatures = pd.read_csv('../train_values_normalized.csv', index_col=0)
y = pd.read_csv('../train_labels.csv', index_col=0)

# choose a selection of features to use in model
useCols = ['resting_blood_pressure','age','chestPain_1','chestPain_2','chestPain_3','chestPain_4']
X = allFeatures[useCols]

# split the train set in train and crossval set
X_train, X_cross, y_train, y_cross = train_test_split(X, y, test_size=0.3, random_state=1)

# apply logistic regression and predict crossval set
logResModel = LogisticRegression(solver='liblinear').fit(X_train, np.ravel(y_train))
prob = logResModel.predict_proba(X_cross)
loss = logLoss(y_cross, prob)[0,1]
print(loss)

