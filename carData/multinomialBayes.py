import numpy as np
import csv
# from sklearn.ensemble import RandomForestClassifier
from bayesClassifier import *
data = []
with open('car.csv', 'rt') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        # Normalize data
        for i in range(len(row)):
            if (row[i] == 'low' or row[i] == 'small'):
                row[i] = 1
            elif (row[i] == 'med'):
                row[i] = 2
            elif (row[i] == 'big' or row[i] == 'high'):
                row[i] = 3
            elif (row[i] == 'vhigh'):
                row[i] = 4
            elif (row[i] == 'more' or row[i] == '5more'):
                row[i] = 5
        data.append(row)
data = np.array(data)
trainData = data[:1600, :6].astype(np.uint8)
trainLabel = data[:1600, 6]
testData = data[1600:, :6].astype(np.uint8)
testLabel = data[1600:, 6]
clf = multinomialBayes(fit_prior = 0.5)
clf.fit(trainData, trainLabel)
print('Score: ' + str(clf.score(testData, testLabel)))
# print('Predict: ' + str(clf.predict(testData)))
# print('Predict Probability: ' + str(clf.predictProbability()))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
