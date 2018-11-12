import numpy as np
import csv
from bayesClassifier import *
data = []
with open('diabetes.csv', 'rb') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        data.append(row)
data = np.array(data)
trainData = data[:600, :8].astype(np.float)
trainLabel = data[:600, 8]
testData = data[600:, :8].astype(np.float)
testLabel = data[600:, 8]
clf = gaussianBayes()
clf.fit(trainData, trainLabel)
print('Score mycode: ' + str(clf.score(testData, testLabel)))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(trainData, trainLabel)
print('Score sklearn: ' + str(clf.score(testData, testLabel)))
# print('Predict: ' + str(clf.predict(testData)))
# print('Predict Probability: ' + str(clf.predictProbability()))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
