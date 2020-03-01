# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore the future warnings while execution

from sklearn import tree
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.naive_bayes import GaussianNB # a Naive-Bayes classifier
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn import metrics # For computing the metrics of the model deployed
from sklearn.model_selection import StratifiedKFold # to ustlise kfold cross validation
from sklearn.preprocessing import Normalizer # used for normalising the different scales of a data
from matplotlib import pyplot as plt

from sklearn.tree import export_graphviz


dataset = pd.read_csv("Problem2_Data.csv") # Reading the dataset

"""Verify is there any null values in the data set provided"""
"""nan = 0
nanSeries = dataset.isnull().any()
nanSeries = nanSeries.to_dict()

for x in nanSeries:
    if nanSeries[x]:
        nan = 1
        break

print(nan)
print(dataset.mean())"""

# Splitting the attributes into independent and dependent attributes
indVar = dataset.iloc[:, 1:-1].values # the set of independent variables on which the model is built
target = dataset.iloc[:,-1].values # the target variable which will give the class they are placed in


# handling the missing data and replace values with nan from numpy and replace with mean of other values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(indVar[:, 1:])
indVar[:, 1:] = imputer.transform(indVar[:, 1:])

# feature scaling the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
scaler = Normalizer()
indVarResc = scaler.fit_transform(indVar)

# blank lists to store the predicted values and actual values
predicted_target = []
expected_target = []

# Set to 10 folds to data
skf = StratifiedKFold(n_splits=10, shuffle=True)

for train_index, test_index in skf.split(indVarResc, target):
    # using array index for np arrays for distributing the data
    indVar_train, indVar_test = indVarResc[train_index], indVarResc[test_index]
    target_train, target_test = target[train_index], target[test_index]
    
    # create and fit classifier
    classifier = GaussianNB()
    classifier.fit(indVar_train, target_train)
    
    # create a random forest classifier
    classifierRF = RandomForestClassifier(n_jobs=2, random_state=0)
    classifierRF.fit(indVar_train, target_train)
    
    # create a Decision Tree Classifier
    classifierDT = tree.DecisionTreeClassifier(n_estimators=10)
    classifierDT.fit(indVar_train, target_train)
    
    # store result from classification
    predicted_target.extend(classifierRF.predict(indVar_test))
    
    # store expected result for this specific fold
    expected_target.extend(target_test)

# save and print accuracy of the model
accuracy = metrics.accuracy_score(expected_target, predicted_target)
classReport = metrics.classification_report(expected_target, predicted_target)
confMatrix = metrics.confusion_matrix(expected_target, predicted_target)
print("Accuracy: " + round(accuracy*100).__str__() + "%")
print("Classification Report: \n" + classReport.__str__())
print("Confusion Matrix:\n" + confMatrix.__str__())

print(classifierRF)

correlations = dataset.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()
# The line/model
# plt.scatter(indVarResc, target)
# plt.xlabel('True Value')
# plt.ylabel('Predictions')
# plt.show()
# export_graphviz(classifierDT, out_file ='tree.dot', feature_names =['Production Cost'])