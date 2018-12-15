import numpy as np
import csv
from bayesClassifier import *
data = []
with open('iris_shuffled.csv', 'rt') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        data.append(row)
data = np.array(data)
trainData = data[:100, :4].astype(np.float)
trainLabel = data[:100, 4]
testData = data[100:, :4].astype(np.float)
testLabel = data[100:, 4]
clf = gaussianBayes()
clf.fit(trainData, trainLabel)
print('Score mycode: ' + str(clf.score(testData, testLabel)))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(trainData, trainLabel)
print('Score sklearn: ' + str(clf.score(testData, testLabel)))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
