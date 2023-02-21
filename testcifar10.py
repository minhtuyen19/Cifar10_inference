# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:13:44 2023

@author: Admin
"""
# =============================================================================

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import onnxmltools
from tensorflow.keras.models import load_model
#Data loading and preprocessing
(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.cifar10.load_data()
classes = ['airplane', 'automobie', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


#one hot coding: 0000000000, 1000000000, 0100000000, 0010000000, 0001000000

Xtrain, Xtest = Xtrain/255, Xtest/255 #chuan hoa du lieu tu 0->1
ytrain, ytest = tf.keras.utils.to_categorical(ytrain), tf.keras.utils.to_categorical(ytest) #one hot and coding label
# =============================================================================
# model_trainning_first = tf.keras.Sequential ([
#     #Conv2D (numofKernel, sizeofKernel, input, relu activation)
#     tf.keras.layers.Conv2D(32,(3,3), input_shape=(32,32,3), activation = 'relu'),
#     #poolingsize  
#     tf.keras.layers.MaxPool2D((2,2)),
#     tf.keras.layers.Dropout(0.15),
# 
#     tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
#     tf.keras.layers.MaxPool2D((2,2)),
#     tf.keras.layers.Dropout(0.20),
# 
#     tf.keras.layers.Conv2D(128,(3,3), activation = 'relu'), 
#     tf.keras.layers.MaxPool2D((2,2)),
#     tf.keras.layers.Dropout(0.20),
# 
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(1000, activation='relu'),
#     tf.keras.layers.Dense(256, activation='relu'),  
#     tf.keras.layers.Dense(10, activation='softmax'),
# ])
# 
# #model_trainning_first.summary()
# 
# model_trainning_first.compile(optimizer='adam', 
#                             loss = 'categorical_crossentropy', 
#                             metrics=['accuracy'])
# model_trainning_first.fit(Xtrain, ytrain, epochs=5, shuffle=True)
# 
# # #Đánh giá model với dữ liệu test set
# score = model_trainning_first.evaluate(Xtest, ytest, verbose=0)
# print(score)
# model_trainning_first.save('modelnew2_cifar10.h5')
# =============================================================================
# ing = cv2.imread("EDG.jpg")
# ing_3D = cv2.resize(ing, (28,28)) #get size 200, 200
# print(ing_3D.shape)
# plt.imshow(ing_3D)
# =============================================================================
class conv2d:
    def __init__(self, inputs, numOfKernel, kernelSize, stride = 1, padding = 0):
        self.kernel = np.random.randn(numOfKernel, kernelSize, kernelSize)
        
        self.inputs = np.pad(inputs, ((0,0),(padding, padding), (padding , padding)),'constant')
        self.stride = stride
       
        h_out = ((self.inputs.shape[0] - self.kernel.shape[1] + 2*padding)//self.stride + 1)
        w_out = ((self.inputs.shape[1] - self.kernel.shape[2] + 2*padding)//self.stride + 1)
        self.results = np.zeros((h_out, w_out, self.kernel.shape[0]))
    def getROI(self):
        for i in range(self.inputs.shape[2]):
            for row in range((self.inputs.shape[0] - self.kernel.shape[1])//self.stride + 1):
                for col in range((self.inputs.shape[1] - self.kernel.shape[2])//self.stride + 1):
                    roi = self.inputs[row*self.stride: row*self.stride + self.kernel.shape[1], col*self.stride : col*self.stride + self.kernel.shape[2], i]
                    yield i, row, col, roi
                    
    def operate(self):
        for layer in range(self.kernel.shape[0]):
            for i, row, col, roi in self.getROI():
                self.results[row, col, layer] = np.sum(roi*self.kernel[layer])
        return self.results
    
class relu:
    def __init__(self, inputs):
        self.inputs = inputs
        self.result = np.zeros((self.inputs.shape[0],
                                self.inputs.shape[1],
                                self.inputs.shape[2]))
    def operate(self):
        for layer in range(self.inputs.shape[2]):
            for row in range(self.inputs.shape[0]):
                for col in range(self.inputs.shape[1]):
                    self.result[row, col, layer] = 0 if self.inputs[row, col, layer] < 0 else self.inputs[row, col, layer]
        return self.result
    
class maxpool2:
    def __init__(self, inputs, pool_size = 2, stride = 2):
        self.inputs = inputs
        self.pool_size = pool_size
        self.stride = stride
        self.result = np.zeros((((self.inputs.shape[0])//self.pool_size),
                               (((self.inputs.shape[1])//self.pool_size)), 
                               self.inputs.shape[2]))
    def operate(self):
        for layer in range (self.inputs.shape[2]):
            for row in range (((self.inputs.shape[0])//self.pool_size)):
                for col in range(((self.inputs.shape[1])//self.pool_size)):
                    self.result[row, col, layer] = (np.max(self.inputs[row*self.stride : row*self.stride + self.pool_size,
                                                               col*self.stride : col*self.stride + self.pool_size, layer]))
        return self.result
    
class flatten:
    def __init__(self, inputs):
        self.inputs = inputs
    def operate(self):
        x = self.inputs.flatten()
        x1 = np.reshape(x, (x.shape[0], 1))
        return x1

class dense:
    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
    def operate(self):
        weight1 = np.random.randn(self.weights, self.inputs.shape[0])
        bias1 = np.zeros((self.bias, 1))
        output = np.dot(weight1, self.inputs) + bias1
        return output

class activation:
    def softmax(X):
        eX = np.exp(X - np.max(X, axis = 0, keepdims= True))
        Z = eX/eX.sum(axis = 0)
        return Z
    def sigmoid(x):
        return 1/(1+np.exp(-x))
        #dao ham ham sigmoid
    def sigmoid_derivative(x):
            return x*(1-x)
    
        
# Input image of size 28x28x1
#input_image = np.random.rand(32, 32, 1)
input_image = Xtest[99].reshape((1,32,32,3))     

def predict(x):  
    l = conv2d(x, 6, 5, stride = 1, padding = 0).operate()
    l = relu(l).operate()
    l = maxpool2(l,pool_size = 2, stride = 2).operate()
    l = conv2d(l, 16, 5, stride = 1, padding = 0).operate()
    l = relu(l).operate()
    l = maxpool2(l, pool_size = 2, stride = 2).operate()
    l = flatten(l).operate()
    l = dense(l, 120, 120).operate()
    l = activation.sigmoid(l)
    l = dense(l, 84, 84).operate()
    l = activation.sigmoid(l)
    l = dense(l, 10,10).operate()
    l = activation.sigmoid(l)
    return l

model = load_model('modelnew3_cifar10.h5')
# =============================================================================
# pre = model.predict((input_image))
# print(classes[np.argmax(pre)]) #lay vi tri xac suat lon nhat
# plt.imshow(Xtest[99])
# plt.show() 
# =============================================================================
np.random.shuffle(Xtest)

acc = 0
for i in range (100):
    plt.subplot(10, 10, i+1)
    plt.imshow(Xtest[500+i]) 
    if (np.argmax(model.predict(Xtest[500+i].reshape((-1,32,32,3)))) == ytest[500+i][0]):
        acc +=1
    plt.title(classes[np.argmax(model.predict(Xtest[500+i].reshape((-1,32,32,3))))])
    plt.axis('off')

plt.show()
print(acc)