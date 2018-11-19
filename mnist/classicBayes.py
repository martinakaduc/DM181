import numpy as np
from bayesClassifier import *
def loadMNIST( prefix):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( prefix + '-images-idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( prefix + '-labels-idx1-ubyte', dtype = 'ubyte' )[2 * intType.itemsize:]

    return data, labels

trainData, trainLabel = loadMNIST( "train")
testData, testLabel = loadMNIST( "t10k")
clf = bayesClassic(fit_prior = 0.5)
clf.fit(trainData, trainLabel)
print('Score: ' + str(clf.score(testData, testLabel)))
# print('Predict: ' + str(clf.predict(testData)))
# print('Predict Probability: ' + str(clf.predictProbability()))
# saveModel(clf, 'model.sav') #save model for future use
# loaded_model = loadModel('model.sav') #load model from file
# print(loaded_model.predict(testData)) #try to predict using loaded_model
