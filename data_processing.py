import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt

''' This script processes the train and test data into all numerical values. '''

rawTrain = pd.read_csv('../train_values.csv', index_col=0)
rawTest = pd.read_csv('../test_values.csv', index_col=0)

# change categorical data into one-hot:
trainSlope_oneHot = pd.get_dummies(rawTrain['slope_of_peak_exercise_st_segment'], prefix='slope')
trainThal_oneHot = pd.get_dummies(rawTrain['thal'])
trainChestPain_oneHot = pd.get_dummies(rawTrain['chest_pain_type'], prefix='chestPain')
trainResting_oneHot = pd.get_dummies(rawTrain['resting_ekg_results'], prefix='restingEkg')
testSlope_oneHot = pd.get_dummies(rawTest['slope_of_peak_exercise_st_segment'], prefix='slope')
testThal_oneHot = pd.get_dummies(rawTest['thal'])
testChestPain_oneHot = pd.get_dummies(rawTest['chest_pain_type'], prefix='chestPain')
testResting_oneHot = pd.get_dummies(rawTest['resting_ekg_results'], prefix='restingEkg')

# replace categorical columns by one-hot
rawTrain.drop(['slope_of_peak_exercise_st_segment','thal','chest_pain_type','resting_ekg_results'], axis=1, inplace=True)
rawTrain = rawTrain.join([trainSlope_oneHot, trainThal_oneHot, trainChestPain_oneHot, trainResting_oneHot])
rawTest.drop(['slope_of_peak_exercise_st_segment','thal','chest_pain_type','resting_ekg_results'], axis=1, inplace=True)
rawTest = rawTest.join([testSlope_oneHot, testThal_oneHot, testChestPain_oneHot, testResting_oneHot])

# check for NaN's in dataset
print(rawTrain.isnull().values.any())
print(rawTest.isnull().values.any())

# apply z-score normalization to numerical data
numCols = ['resting_blood_pressure', 'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'age', 'max_heart_rate_achieved']
for col in numCols:
	rawTest[col] = (rawTest[col] - rawTrain[col].mean()) / rawTrain[col].std()
	rawTrain[col] = (rawTrain[col] - rawTrain[col].mean()) / rawTrain[col].std()
	print(rawTrain[col].mean(), rawTrain[col].std()) # should be 0 and 1

# Sanity check
plt.figure(1)
boxplot = rawTrain.boxplot(column=numCols)
plt.figure(2)
boxplot = rawTest.boxplot(column=numCols)

# Storing processed data into new file
rawTrain.to_csv('../train_values_normalized.csv')
rawTest.to_csv('../test_values_normalized.csv')

plt.show()