# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:06:26 2023

@author: Admin
"""

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# 2. Load dữ liệu MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape lai du lieu cho dung kich thuoc ma keras yeu cau

X_test = X_test.reshape(X_test.shape[0], 1,28,28,1)

# One hot encoding label (Y)
Y_test = np_utils.to_categorical(y_test, 10)

# =============================================================================
# (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
# classes = ['airplane', 'automobie', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# # 
# # 
# # #one hot coding: 0000000000, 1000000000, 0100000000, 0010000000, 0001000000
# # 
# X_train, X_test = X_train/255, X_test/255 #chuan hoa du lieu tu 0->1
# Y_train, Y_test = tf.keras.utils.to_categorical(Y_train), tf.keras.utils.to_categorical(Y_test) #one hot and coding label
# 
# =============================================================================
# =============================================================================
# class conv2d:
#     def __init__(self, inputs, numOfKernel, kernelSize, stride = 1, padding = 0):
#         self.kernel = np.random.randn(numOfKernel, kernelSize, kernelSize)
#         
#         self.inputs = np.pad(inputs, ((0,0),(padding, padding), (padding , padding)),'constant')
#         self.stride = stride
#        
#         h_out = ((self.inputs.shape[0] - self.kernel.shape[1] + 2*padding)//self.stride + 1)
#         w_out = ((self.inputs.shape[1] - self.kernel.shape[2] + 2*padding)//self.stride + 1)
#         self.results = np.zeros((h_out, w_out, self.kernel.shape[0]))
#     def getROI(self):
#         for i in range(self.inputs.shape[2]):
#             for row in range((self.inputs.shape[0] - self.kernel.shape[1])//self.stride + 1):
#                 for col in range((self.inputs.shape[1] - self.kernel.shape[2])//self.stride + 1):
#                     roi = self.inputs[row*self.stride: row*self.stride + self.kernel.shape[1], col*self.stride : col*self.stride + self.kernel.shape[2], i]
#                     yield i, row, col, roi
#                     
#     def operate(self):
#         for layer in range(self.kernel.shape[0]):
#             for i, row, col, roi in self.getROI():
#                 self.results[row, col, layer] = np.sum(roi*self.kernel[layer])
#         return self.results
#     
# class relu:
#     def __init__(self, inputs):
#         self.inputs = inputs
#         self.result = np.zeros((self.inputs.shape[0],
#                                 self.inputs.shape[1],
#                                 self.inputs.shape[2]))
#     def operate(self):
#         for layer in range(self.inputs.shape[2]):
#             for row in range(self.inputs.shape[0]):
#                 for col in range(self.inputs.shape[1]):
#                     self.result[row, col, layer] = 0 if self.inputs[row, col, layer] < 0 else self.inputs[row, col, layer]
#         return self.result
#     
# class maxpool2:
#     def __init__(self, inputs, pool_size = 2, stride = 2):
#         self.inputs = inputs
#         self.pool_size = pool_size
#         self.stride = stride
#         self.result = np.zeros((((self.inputs.shape[0])//self.pool_size),
#                                (((self.inputs.shape[1])//self.pool_size)), 
#                                self.inputs.shape[2]))
#     def operate(self):
#         for layer in range (self.inputs.shape[2]):
#             for row in range (((self.inputs.shape[0])//self.pool_size)):
#                 for col in range(((self.inputs.shape[1])//self.pool_size)):
#                     self.result[row, col, layer] = (np.max(self.inputs[row*self.stride : row*self.stride + self.pool_size,
#                                                                col*self.stride : col*self.stride + self.pool_size, layer]))
#         return self.result
#     
# class flatten:
#     def __init__(self, inputs):
#         self.inputs = inputs
#     def operate(self):
#         x = self.inputs.flatten()
#         x1 = np.reshape(x, (x.shape[0], 1))
#         return x1
# 
# class dense:
#     def __init__(self, inputs, weights, bias):
#         self.inputs = inputs
#         self.weights = weights
#         self.bias = bias
#     def operate(self):
#         weight1 = np.random.randn(self.weights, self.inputs.shape[0])
#         bias1 = np.zeros((self.bias, 1))
#         output = np.dot(weight1, self.inputs) + bias1
#         return output
# 
# class activation:
#     def softmax(X):
#         eX = np.exp(X - np.max(X, axis = 0, keepdims= True))
#         Z = eX/eX.sum(axis = 0)
#         return Z
#     def sigmoid(x):
#         return 1/(1+np.exp(-x))
#         #dao ham ham sigmoid
#     def sigmoid_derivative(x):
#             return x*(1-x)
# =============================================================================
    
        
# Input image of size 28x28x1
#input_image = np.random.rand(32, 32, 1)
#input_image = Xtest[103].reshape((32,32,3))     


class conv2d:
    def __init__(self, inputs, kernel, bias, stride = 1, padding = 0):
        self.kernel = kernel
        self.bias = bias
        self.stride = stride
        self.inputs = inputs
        
        n_inputs, h_in, w_in, n_filters = self.inputs.shape
        h_weights, w_weights, _, n_out = self.kernel.shape
        
        h_out = (self.inputs.shape[1] - self.kernel.shape[0] + 2 * padding) // self.stride + 1
        w_out = (self.inputs.shape[2] - self.kernel.shape[1] + 2 * padding) // self.stride + 1
        
        self.inputs = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
        self.outputs = np.zeros((self.inputs.shape[0], h_out, w_out, self.kernel.shape[3]))
        print('results: ', self.outputs.shape)
    def operate(self):
        for i in range(self.inputs.shape[0]):
            for h in range((self.inputs.shape[1] - self.kernel.shape[0]) // self.stride + 1):
                for w in range((self.inputs.shape[2] - self.kernel.shape[1]) // self.stride + 1):
                    for c in range(self.kernel.shape[3]):
                        self.outputs[i, h, w, c] = np.sum(self.inputs[i, h*self.stride:h*self.stride+self.kernel.shape[0], w*self.stride:w*self.stride+self.kernel.shape[1], :] * self.kernel[:, :, :, c]) + self.bias[c]
        return self.outputs
            
   
class maxpool2:
    def __init__(self, inputs, pool_size = 2, stride = 2):
        self.inputs = inputs
        self.pool_size = pool_size
        self.stride = stride
        n_inputs, h_in, w_in, n_filters = self.inputs.shape
        
        h_out = (self.inputs.shape[1] - self.pool_size) // self.stride + 1
        w_out = (self.inputs.shape[2] - self.pool_size) // self.stride + 1
        
        self.outputs = np.zeros((self.inputs.shape[0], h_out, w_out, self.inputs.shape[3]))
    def operate(self):
        for i in range(self.inputs.shape[0]):
            for h in range((self.inputs.shape[1] - self.pool_size) // self.stride + 1):
                for w in range((self.inputs.shape[2] - self.pool_size) // self.stride + 1):
                    for c in range(self.inputs.shape[3]):
                        self.outputs[i, h, w, c] = np.max(self.inputs[i, h*self.stride:h*self.stride+self.pool_size, w*self.stride:w*self.stride+self.pool_size, c])
                        
        return self.outputs
    
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
        #weight1 = np.random.randn(self.weights, self.inputs.shape[0])
        #bias1 = np.zeros((self.bias, 1))
        self.weights = self.weights.reshape(self.weights.shape[0], self.weights.shape[1])
        print('weights: ', self.weights.shape)
        self.bias = np.reshape(self.bias, (1, self.bias.shape[0]))
        print('bias: ', self.bias.shape)
        output = np.dot(self.inputs, self.weights) + self.bias
        return output

class activation:
    def softmax(x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
    def sigmoid(x):
        return 1/(1+np.exp(-x))
        #dao ham ham sigmoid
    def sigmoid_derivative(x):
        return x*(1-x)

model = keras.models.load_model('model_mnist.h5')

#model.summary()
weights_1 = np.array(model.layers[0].get_weights(), dtype=object)
W_1 = weights_1[0].reshape(3, 3, 1, 32)
#print('W_1: ', weights_1[0].shape)
b_1 = weights_1[1]
#print(b_1.shape)

weights_2 = np.array(model.layers[1].get_weights(), dtype=object)
layer_name = model.layers[1].name

#print(weights_2[0][].shape)
# =============================================================================
W_2 = weights_2[0]
#W_2 = W_2[:, :, 0, :].reshape(32, 3, 3)
print(W_2.shape)
b_2 = weights_2[1]

weights_3 = np.array(model.layers[4].get_weights(), dtype=object)
W_3 = weights_3[0]
b_3 = weights_3[1]
#layer_name1 = model.layers[4].name

#print(layer_name1)
#print(W_3.shape)
#print(b_3.shape)

weights_4 = np.array(model.layers[5].get_weights(), dtype=object)
W_4 = weights_4[0]
b_4 = weights_4[1]



# 
l1 = conv2d(X_test[101], W_1, b_1, 1,0).operate()
l1 = activation.sigmoid(l1)
l2 = conv2d(l1, W_2, b_2, 1, 0).operate()
l2 = activation.sigmoid(l2)
#print(l2)
l3 = maxpool2(l2,pool_size = 2, stride = 2).operate()
l4 = flatten(l3).operate()
l4 = l4.reshape(1, l4.shape[0])
l5 = dense(l4, W_3, b_3).operate()
l6 = activation.sigmoid(l5)
#print('l6: ', l6)
l7 = dense(l6, W_4, b_4).operate()
l8 = activation.softmax(l7)

print('Gia tri du doan: ', np.argmax(l7))

plt.imshow(X_test[101].reshape(28,28), cmap='gray')

# 
# =============================================================================





