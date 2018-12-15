from __future__ import division
import numpy as np
import csv
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

data = []
with open('diabetes.csv', 'rt') as csvfile:
    csvObj = csv.reader(csvfile, delimiter=',')
    for row in csvObj:
        data.append(row)
data = np.array(data)
trainData = data[:600, :8]
trainLabel = data[:600, 8].reshape(-1,1).astype(np.uint8)
testData = data[600:, :8]
testLabel = data[600:, 8].astype(np.uint8)
# from keras.utils import to_categorical
trainLabel = keras.utils.to_categorical(trainLabel, num_classes=2)
model = Sequential()

model.add(Dense(512, activation='relu', input_dim = 8))
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
a = []

for X in testData:
    classes = model.predict(X.reshape(-1,8))
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
