from __future__ import division
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

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
trainLabel = keras.utils.to_categorical(trainLabel, num_classes=10)
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(trainData, trainLabel, epochs=100, batch_size=100)
a = []
for X in testData:
    classes = model.predict(X.reshape(-1,28,28))
    a.append(np.argmax(classes[0]))
count = 0
for i in range(len(a)):
    if(a[i] == testLabel[i]):
        count += 1
print('Score: ' + str(count/len(a)))

# # serialize model to YAML
# model_yaml = model.to_yaml()
# with open("model.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# from keras.models import model_from_yaml
# # load YAML and create model
# yaml_file = open('model.yaml', 'r')
# loaded_model_yaml = yaml_file.read()
# yaml_file.close()
# loaded_model = model_from_yaml(loaded_model_yaml)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
