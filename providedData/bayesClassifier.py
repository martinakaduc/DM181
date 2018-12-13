# Written by Nguyen Quang Duc
# Please do not steal my code
from __future__ import division
import numpy as np
import pickle

class bayesClassic(object):
    def __init__(self, fit_prior = 1):
        self.fit_prior = fit_prior #priori estimate
        self.y_value = []
        self.y_proba = []
        self.X_value = []
        self.X_proba = []

    def fit(self, X, y):
        self.X_train = np.array([x.flatten() for x in X])
        self.y_train = y
        self.sample_size = self.X_train.shape[-1] #equivalent sample size (column:-1, row:0)

        #calculate probability of each y (P(y))
        print('Calculating labels\'s probability')
        for y in range(self.y_train.shape[0]):
            if (self.y_train[y] not in self.y_value):
                self.y_value.append(self.y_train[y])
                self.y_proba.append(0)
            self.y_proba[self.y_value.index(self.y_train[y])] += 1
            print('Process: %s' % (100*(y+1)/self.y_train.shape[0]))
        self.y_proba = [k / sum(self.y_proba) for k in self.y_proba]

        #calculate probability of each feature if each y (P((a-X) | y))
        print('Calculating each feature\'s probability')
        percent = 0
        for y in range(len(self.y_value)):
            self.X_proba.append([])
            for X_v in range(self.sample_size):
                self.X_valueEachFeature = []
                self.X_count = []
                for X_h in range(self.X_train.shape[0]):
                    if (self.X_train[X_h,X_v] not in self.X_valueEachFeature):
                        self.X_valueEachFeature.append(self.X_train[X_h,X_v])
                        self.X_count.append(0)
                    if (self.y_train[X_h] == self.y_value[y]):
                        self.X_count[self.X_valueEachFeature.index(self.X_train[X_h,X_v])] += 1
                    percent += (100 / (len(self.y_value) * self.sample_size * self.X_train.shape[0]))
                    print('Process: %s' % percent)
                self.X_proba[y].append([(X + self.fit_prior*self.sample_size) / (self.sample_size + self.y_proba[y]*self.y_train.shape[-1])
                                for X in self.X_count])
                if (y == 0):
                    self.X_value.append(self.X_valueEachFeature)

    def predict(self, X_predict):
        self.X_predict = X_predict.flatten()
        self.y_predict = []
        for y in range(len(self.y_value)):
            self.y_predict.append(self.y_proba[y])
            for X in range(self.X_predict.shape[-1]):
                if (self.X_predict[X] in self.X_value[X]):
                    self.y_predict[y] *= self.X_proba[y][X][self.X_value[X].index(self.X_predict[X])]
        return (self.y_value[np.argmax(self.y_predict)])

    def predictProbability(self):
        return (self.y_predict[np.argmax(self.y_predict)]/sum(self.y_predict))

    def score(self, X, y):
        self.X_test = X
        self.y_test = y
        validCount = 0
        for i in range(self.y_test.shape[-1]):
            if (self.predict(self.X_test[i].reshape(1,-1)) == self.y_test[i]):
                validCount += 1
        return (validCount / self.y_test.shape[-1])

def saveModel(clf, filename):
    pickle.dump(clf, open(filename, 'wb'))

def loadModel(filename):
    return (pickle.load(open(filename, 'rb')))

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def variance(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / (len(numbers) - 1)
    return variance

def standardDeviation(numbers):
    return np.sqrt(variance(numbers))

class gaussianBayes(bayesClassic):
    def __init__(self):
        self.y_value = []
        self.X_mean = []
        self.X_stDev = []
        self.y_proba = []

    def fit(self, X, y):
        self.X_train = np.array([x.flatten() for x in X])
        self.y_train = y

        #calculate probability of each y (P(y))
        print('Calculating labels\'s probability')
        for y in range(self.y_train.shape[0]):
            if (self.y_train[y] not in self.y_value):
                self.y_value.append(self.y_train[y])
                self.y_proba.append(0)
            self.y_proba[self.y_value.index(self.y_train[y])] += 1
            print('Process: %s' % (100*(y+1)/self.y_train.shape[0]))
        self.y_proba = [k / sum(self.y_proba) for k in self.y_proba]
        #calculate probability of each feature if each y (P((a-X) | y))
        print('Calculating each feature\'s probability')
        percent = 0
        for y in range(len(self.y_value)):
            self.X_mean.append([])
            self.X_stDev.append([])
            for X_v in range(self.X_train.shape[-1]):
                self.X_trainEachY = []
                for X_h in range(self.X_train.shape[0]):
                    if (self.y_train[X_h] == self.y_value[y]):
                        self.X_trainEachY.append(self.X_train[X_h,X_v])
                    percent += (100 / (len(self.y_value) * self.X_train.shape[-1] * self.X_train.shape[0]))
                    print('Process: %s' % percent)
                self.X_mean[y].append(mean(self.X_trainEachY))
                self.X_stDev[y].append(standardDeviation(self.X_trainEachY))

    def predict(self, X_predict):
        self.X_predict = X_predict.flatten()
        self.y_predict = []
        for y in range(len(self.y_value)):
            self.y_predict.append(self.y_proba[y])
            for X in range(self.X_predict.shape[-1]):
                if (self.X_stDev[y][X] != 0):
                    self.y_predict[y] *= (np.exp(-(self.X_predict[X] - self.X_mean[y][X])**2 / (2*self.X_stDev[y][X]**2)) / (np.sqrt(2*np.pi)*self.X_stDev[y][X]))
                else:
                    if ((self.X_predict[X] - self.X_mean[y][X]) == 0):
                        self.y_predict[y] *= self.X_train.shape[0]
                    else:
                        self.y_predict[y] *= 0
        return (self.y_value[np.argmax(self.y_predict)])

class multinomialBayes(bayesClassic):
    def fit(self, X, y):
        self.X_train = np.array([x.flatten() for x in X])
        self.y_train = y
        self.sample_size = self.X_train.shape[-1] #equivalent sample size (column:-1, row:0)

        #calculate probability of each y (P(y))
        print('Calculating labels\'s probability')
        for y in range(self.y_train.shape[0]):
            if (self.y_train[y] not in self.y_value):
                self.y_value.append(self.y_train[y])
                self.y_proba.append(0)
            self.y_proba[self.y_value.index(self.y_train[y])] += 1
            print('Process: %s' % (100*(y+1)/self.y_train.shape[0]))
        self.y_proba = [k / sum(self.y_proba) for k in self.y_proba]

        #calculate probability of each feature if each y (P((a-X) | y))
        print('Calculating each feature\'s probability')
        percent = 0
        for y in range(len(self.y_value)):
            self.X_proba.append([])
            for X_v in range(self.sample_size):
                self.X_valueEachFeature = []
                self.X_count = []
                for X_h in range(self.X_train.shape[0]):
                    if (self.X_train[X_h,X_v] not in self.X_valueEachFeature):
                        self.X_valueEachFeature.append(self.X_train[X_h,X_v])
                        self.X_count.append(0)
                    if (self.y_train[X_h] == self.y_value[y]):
                        self.X_count[self.X_valueEachFeature.index(self.X_train[X_h,X_v])] += 1
                    percent += (100 / (len(self.y_value) * self.sample_size * self.X_train.shape[0]))
                    print('Process: %s' % percent)
                self.X_proba[y].append([(X + self.fit_prior) / (self.fit_prior*self.sample_size + self.y_proba[y]*self.y_train.shape[-1])
                                for X in self.X_count])
                if (y == 0):
                    self.X_value.append(self.X_valueEachFeature)
