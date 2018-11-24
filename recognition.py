import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import h5py


# Reading Dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dct = unpickle("data_batch_1")
X = dct[b'data']
Y = np.array(dct[b'labels'])

for i in range(2,5):
    fn = "data_batch_" +str(i)
    dct = unpickle(fn)
    xTrain = np.vstack((X, dct[b'data']))
    yTrain = np.concatenate((Y, dct[b'labels']))


dct = unpickle("data_batch_5")
xValidate = dct[b'data']
yValidate = np.array(dct[b'labels'])

dct = unpickle("test_batch")
xTest = dct[b'data']
yTest = np.array(dct[b'labels'])


# Defining Models
model1 = Sequential([
    Dense(50, input_dim=3072),
    Activation('relu'),
    BatchNormalization(),
    Dense(10),
    Activation('softmax')
])
model2 = Sequential([
    Conv2D(filters=32, kernel_size=(2,2), strides=1, activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(50),
    Activation('relu'),
    BatchNormalization(),
    Dense(10),
    Activation('softmax')
])
model3 = Sequential([
    Conv2D(filters=64, kernel_size=(4,4), strides=1, activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(50),
    Activation('relu'),
    BatchNormalization(),
    Dense(10),
    Activation('softmax')
])

nadam = optimizers.Nadam()
model1.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])
model3.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])

# one-hot-encoding the classes.
yTrainHot = to_categorical(yTrain, num_classes=10)
yValidateHot = to_categorical(yValidate, num_classes=10)
yTestHot = to_categorical(yTest, num_classes=10)

# defining the stopping criteria.
esc = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')


# model 1
cp = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
m1 = model1.fit(xTrain, yTrainHot, epochs=1000, callbacks=[esc, cp], validation_data=(xValidate, yValidateHot))

# model 2
xTrain = xTrain.reshape(xTrain.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
xValidate = xValidate.reshape(xValidate.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
cp = ModelCheckpoint(filepath='weights1.hdf5', verbose=1, save_best_only=True)
m2 = model2.fit(xTrain, yTrainHot, epochs=1000, callbacks=[esc, cp], validation_data=(xValidate, yValidateHot))

# model 3
cp = ModelCheckpoint(filepath='weights2.hdf5', verbose=1, save_best_only=True)
m3 = model3.fit(xTrain, yTrainHot, epochs=1000, callbacks=[esc, cp], validation_data=(xValidate, yValidateHot))


# Testing with best weights learned during training.

model1.load_weights('weights.hdf5')
m1Score = model1.evaluate(xTest, yTestHot)

xTest = xTest.reshape(xTest.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
model2.load_weights('weights1.hdf5')
m2Score = model2.evaluate(xTest, yTestHot)

model3.load_weights('weights2.hdf5')
m3Score = model3.evaluate(xTest, yTestHot)


# Loss and accuracy for each model

print(m1Score)
print(m2Score)
print(m3Score)


'''
This tech report (Chapter 3) describes the dataset and the methodology followed when collecting it in much greater detail. 
Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

'''
