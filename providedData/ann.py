from __future__ import division
import numpy as np
import csv
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

data = []
with open('bayesData.csv', 'rt') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        for i in range(len(row)):
            if (row[i] == 'Red' or row[i] == 'Sport' or row[i] == 'Domestic' or row[i] == 'Male'):
                row[i] = 1
            elif (row[i] == 'Yellow' or row[i] == 'Travel' or row[i] == 'Import' or row[i] == 'Female'):
                row[i] = 0
        data.append(row)
data = np.array(data)
trainData = data[:10, :3]
trainLabel = data[:10, 3].reshape(-1,1).astype(np.uint8)
testData = data[10:, :3]
# from keras.utils import to_categorical
trainLabel = keras.utils.to_categorical(trainLabel, num_classes=2)
model = Sequential()

model.add(Dense(512, activation='relu', input_dim = 3))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(trainData, trainLabel, epochs=1000, batch_size=100)
if (np.argmax(model.predict(testData.reshape(-1,3))) == 0):
    predict = 'Female'
else:
    predict = 'Male'
print('Predict: ' + predict)

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
