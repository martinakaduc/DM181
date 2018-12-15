import numpy as np
import csv
# from sklearn.ensemble import RandomForestClassifier
from bayesClassifier import *
data = []
with open('bayesData.csv', 'rt') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        data.append(row)
data = np.array(data)
trainData = data[:10, :3].reshape(-1, 3)
trainLabel = data[:10, 3]
testData = data[10, :3].reshape(-1, 3)
clf = multinomialBayes(fit_prior = 0.5)
clf.fit(trainData, trainLabel)
print('Predict: ' + str(clf.predict(testData)))
print('Predict Probability: ' + str(clf.predictProbability()))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
