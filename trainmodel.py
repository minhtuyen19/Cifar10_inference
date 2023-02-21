import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from tensorflow .keras.optimizers import SGD
#Data loading and preprocessing
(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.cifar10.load_data()
classes = ['airplane', 'automobie', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


#one hot coding: 0000000000, 1000000000, 0100000000, 0010000000, 0001000000

Xtrain, Xtest = Xtrain/255, Xtest/255 #chuan hoa du lieu tu 0->1
ytrain, ytest = tf.keras.utils.to_categorical(ytrain), tf.keras.utils.to_categorical(ytest) #one hot and coding label
model_trainning_first = tf.keras.Sequential ([
    #Conv2D (numofKernel, sizeofKernel, input, relu activation)
    tf.keras.layers.Conv2D(32,(3,3), input_shape=(32,32,3), activation = 'relu'),
    #poolingsize  
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Conv2D(128,(3,3), activation = 'relu'), 
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),  
    tf.keras.layers.Dense(10, activation='softmax'),
])

model_trainning_first.summary()
# opt = SGD(lr=0.001, momentum=0.9)
# model_trainning_first.compile(optimizer='adam', 
#                             loss = 'categorical_crossentropy', 
#                             metrics=['accuracy'])

# model_trainning_first.fit(Xtrain, ytrain, epochs=100, batch_size=64, shuffle=True,
#                           validation_data=(Xtest, ytest), verbose=1)

# # #Đánh giá model với dữ liệu test set
# # score = model_trainning_first.evaluate(Xtest, ytest, verbose=0)
# # print(score)

# model_trainning_first.save('modelnew3_cifar10.h5')