import numpy as np
import csv
from bayesClassifier import *
data = []
with open('iris_shuffled.csv', 'rb') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        data.append(row)
data = np.array(data)
trainData = data[:100, :4]
trainLabel = data[:100, 4]
testData = data[100:, :4]
testLabel = data[100:, 4]
clf = multinomialBayes(fit_prior = 0.5)
clf.fit(trainData, trainLabel)
print('Score: ' + str(clf.score(testData, testLabel)))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
