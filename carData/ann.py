from __future__ import division
import numpy as np
import csv
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

data = []
with open('car.csv', 'rt') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        for i in range(len(row)):
            if (row[i] == 'unacc' or row[i] == 'low' or row[i] == 'small'):
                row[i] = 0
            elif (row[i] == 'acc' or row[i] == 'med'):
                row[i] = 1
            elif (row[i] == 'good' or row[i] == 'big' or row[i] == 'high'):
                row[i] = 2
            elif (row[i] == 'vgood' or row[i] == 'vhigh'):
                row[i] = 3
            elif (row[i] == 'more' or row[i] == '5more'):
                row[i] = 5
        data.append(row)
data = np.array(data)
trainData = data[:1600, :6]
trainLabel = data[:1600, 6].reshape(-1,1).astype(np.uint8)
testData = data[1600:, :6]
testLabel = data[1600:, 6].astype(np.uint8)
# from keras.utils import to_categorical
trainLabel = keras.utils.to_categorical(trainLabel, num_classes=4)
model = Sequential()

model.add(Dense(512, activation='relu', input_dim = 6))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(trainData, trainLabel, epochs=1000, batch_size=100)
a = []
for X in testData:
    classes = model.predict(X.reshape(-1,6))
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
