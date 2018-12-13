import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
data = []
with open('car.csv', 'rb') as csvfile:
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
trainData = data[:1600, :6]
trainLabel = data[:1600, 6]
testData = data[1600:, :6]
testLabel = data[1600:, 6]
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(trainData, trainLabel)
print('Score: ' + str(clf.score(testData, testLabel)))
# print('Predict: ' + str(clf.predict(testData)))
# from bayesClassifier import *
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
