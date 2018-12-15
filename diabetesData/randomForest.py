import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
data = []
with open('diabetes.csv', 'rt') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        data.append(row)
data = np.array(data)
trainData = data[:600, :8]
trainLabel = data[:600, 8]
testData = data[600:, :8]
testLabel = data[600:, 8]
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(trainData, trainLabel)
print('Score: ' + str(clf.score(testData, testLabel)))
# print('Predict: ' + str(clf.predict(testData)))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
