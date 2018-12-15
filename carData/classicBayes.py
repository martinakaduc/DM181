import numpy as np
import csv
# from sklearn.ensemble import RandomForestClassifier
from bayesClassifier import *
data = []
with open('car.csv', 'rt') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        data.append(row)
data = np.array(data)
print(data)
trainData = data[:1600, :6]
trainLabel = data[:1600, 6]
testData = data[1600:, :6]
testLabel = data[1600:, 6]
clf = bayesClassic(fit_prior = 0.5)
clf.fit(trainData, trainLabel)
print('Score: ' + str(clf.score(testData, testLabel)))
# print('Predict: ' + str(clf.predict(testData)))
# print('Predict Probability: ' + str(clf.predictProbability()))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
