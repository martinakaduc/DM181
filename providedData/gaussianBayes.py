import numpy as np
import csv
from bayesClassifier import *
data = []
with open('bayesData.csv', 'rb') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        # Normalize data
        for i in range(len(row)):
            if (row[i] == 'Red' or row[i] == 'Sport' or row[i] == 'Domestic'):
                row[i] = 1
            elif (row[i] == 'Yellow' or row[i] == 'Travel' or row[i] == 'Import'):
                row[i] = 0
        data.append(row)
data = np.array(data)
print(data)
trainData = data[:10, :3].reshape(-1, 3).astype(np.float)
trainLabel = data[:10, 3]
testData = data[10, :3].reshape(-1, 3).astype(np.float)
clf = gaussianBayes()
clf.fit(trainData, trainLabel)
print('Predict: ' + str(clf.predict(testData)))
print('Predict Probability: ' + str(clf.predictProbability()))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
