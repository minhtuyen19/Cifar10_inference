import json
from tensorflow import keras
from keras.datasets import cifar10
from keras import datasets
from keras.utils import np_utils
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Creating a list of all the class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

x_test = x_test[:4]  # Lấy ra 15 phần tử đầu tiên

# a = np.zeros((3, 3, 32, 32))
# print(a)

# a = np.array([[[1,2,3, 6], [3, 4, 5, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]])
# print(a.shape)
# b = np.array([
#         [
#             [
#                 [-0.08550391, -0.1535698 ,  0.2669411 , -0.01623411,
#                 -0.14666736, -0.10674493,  0.00573236,  0.10753579,
#                 0.09180948,  0.03950169,  0.20269449, -0.18005046,
#                 -0.0802247 , -0.3128643 , -0.02132467,  0.04170604,
#                 -0.2952023 , -0.09925324, -0.08406706,  0.15266325,
#                 -0.0306061 ,  0.17817482,  0.06883515, -0.18796954,
#                 -0.19435473,  0.18106896, -0.2104608 , -0.07451914,
#                 -0.04792557,  0.04825107, -0.14031062, -0.19321825]],

#             [
#                 [-0.01211061, -0.12611197,  0.21655753, -0.02701976,
#                 0.13745295, -0.07601351,  0.08040713,  0.00831813,
#                 -0.28707772, -0.1165877 , -0.01835505,  0.01659939,
#                 0.1451141 , -0.24957179, -0.07491435,  0.03234421,
#                 -0.0198013 ,  0.09011956, -0.12485117, -0.08653998,
#                 -0.15746787,  0.02312584, -0.14250846, -0.21831451,
#                 -0.20085745,  0.1275233 , -0.11184104,  0.10147379,
#                 -0.12805001, -0.06650871, -0.02590235, -0.01392371]],

#             [
#                 [-0.22359551,  0.15297773, -0.05511809, -0.16535816,
#                 -0.0577472 ,  0.10508634, -0.19630904, -0.03099463,
#                 0.002953  , -0.03485528,  0.10810248, -0.22648418,
#                 0.04392023, -0.08579255,  0.26836827,  0.08441547,
#                 -0.02018317, -0.11153898, -0.09422036,  0.03794131,
#                 -0.0642808 , -0.10246027, -0.13400441,  0.05489596,
#                 -0.26100725,  0.1007211 , -0.19410598,  0.07738852,
#                 0.23897646, -0.18535678, -0.14631695,  0.25881442]]],


#        [
#             [
#                 [ 0.10011382, -0.0111023 ,  0.01169159, -0.20278792,
#                 -0.18003611,  0.20751141,  0.10582726,  0.17987931,
#                 -0.14770214, -0.04052011, -0.12125973,  0.11295915,
#                 -0.2304624 ,  0.05882748, -0.03988193, -0.27001205,
#                 0.01250037,  0.03486075,  0.19179732,  0.22709362,
#                 0.01457667,  0.18360738,  0.05375492, -0.06115323,
#                 0.01195101,  0.11833888, -0.062857  , -0.3212244 ,
#                 -0.08505782,  0.11688109,  0.03030909,  0.07177468]],

#             [
#                 [ 0.07503512,  0.00347428, -0.00295229, -0.22506084,
#                 -0.18752436,  0.28541997, -0.01336703, -0.14403698,
#                 -0.18105814, -0.00759195, -0.12264832,  0.02750711,
#                 -0.21771768,  0.04092501,  0.2704401 ,  0.13055958,
#                 -0.13229255, -0.06225908,  0.04623315,  0.03779622,
#                 -0.19357862,  0.0418295 , -0.29517943,  0.08237901,
#                 0.04863492, -0.13980351,  0.08651962, -0.02925733,
#                 -0.07860287, -0.10619569, -0.00640487,  0.24794851]],

#             [
#                 [-0.14453462,  0.26763648, -0.07516559, -0.02519073,
#                 -0.2936124 ,  0.16824853, -0.3081435 , -0.03470551,
#                 0.00611225,  0.02715123,  0.15631716, -0.02340811,
#                 -0.19041307, -0.26274714,  0.06457014,  0.01349079,
#                 -0.23067543, -0.26502505, -0.08477856, -0.01753622,
#                 0.18189417, -0.03660496,  0.08983919,  0.21060278,
#                 0.05863881, -0.13863632,  0.0951548 , -0.12700942,
#                 0.2941329 , -0.28717715, -0.09749086, -0.09143655]]],


#         [
#             [[ 0.12754782, -0.15907834, -0.07494561, -0.2178829 ,
#             0.1431979 , -0.03516714,  0.204131  ,  0.09643317,
#             -0.17492466,  0.28231105, -0.1918526 ,  0.17213142,
#             0.12476557,  0.07344089,  0.15888554, -0.23277833,
#             0.08431047,  0.2410253 ,  0.255403  ,  0.2532365 ,
#             0.28380543,  0.35450467, -0.24603531,  0.10514291,
#             0.09369562, -0.06083914,  0.17483675,  0.06148987,
#             -0.05426047,  0.27778098,  0.18834148,  0.10165368]],

#             [[ 0.27627426, -0.00716311,  0.03159506,  0.0448502 ,
#             0.06837387, -0.19217665, -0.06416696, -0.17670436,
#             0.01482319,  0.19484767, -0.15942082, -0.00120893,
#             0.15401357,  0.10936944, -0.09715132, -0.20728534,
#             0.18061632,  0.17687315,  0.0620577 ,  0.09469469,
#             -0.0027799 ,  0.05188345,  0.0246656 , -0.07033475,
#             0.16564144, -0.05589858,  0.11198618, -0.20984796,
#             0.04846259, -0.02679178,  0.15754695, -0.1216803 ]],

#             [[ 0.04924183,  0.14010707, -0.21931085,  0.13093337,
#             -0.10551344, -0.07386293, -0.20474029, -0.2892023 ,
#             0.19488232,  0.26862457,  0.08402619,  0.28121004,
#             -0.02339406, -0.03719459, -0.11932255, -0.23844974,
#             -0.06180196, -0.14092723,  0.09768806, -0.17903586,
#             -0.04789264, -0.11112546,  0.1643823 ,  0.21164815,
#             0.10473405, -0.19292523,  0.19116719, -0.10269222,
#             -0.05080022,  0.0703335 ,  0.29225606, -0.11222625]]
#         ]
#     ])


model = keras.models.load_model('model_mnist.h5')
#model.summary()
# model.summary()

# for i in range(23):
#     weights = model.layers[i].get_weights()
#     layer_name = model.layers[i].name
#     print('Layer name: ', layer_name)
#     print('Layer weights: ', weights)



weights = np.array(model.layers[1].get_weights())
# # layer_name = model.layers[2].name

# # print("Layer name:", layer_name)
print("Layer weights:", weights[0].shape)

# write to txt file
#===========================================================================================
# with open('D:/ASSET/NAM4HK2/Predict/Readmodel/readmnist.txt', 'w') as f:
#     for i in range(6):
#         weights = model.layers[i].get_weights()
#         #layer_name = model.layers[i].name
#         # f.write(layer_name + '\n')
#         f.write(str(weights) + '\n')
    

#============================================================================================
# weights = model.get_weights()
# first_layer_weights = weights[0][1]
# #second_layer_weights = weights[1]
# print(first_layer_weights)
#print('second: ',second_layer_weights)
# model.summary()

# x_test = x_test.astype('float32')
# x_test = x_test/255.0


# model_file = h5py.File('cifar10_classifier1.h5', 'r')
# model_weights = []
# optimizer_weights = []

# for layer in model_file['model_weights']:
#     w1=model_weights.append(layer[()])
#     print(w1)
    
# for layer in model_file['optimizer_weights']:
#     w2=optimizer_weights.append(layer[()])
#     print(w2)
#===================================================
# model_weights = weights_file['model_weights']
# layer_names = [name for name in model_weights]
# print(layer_names)

# # Load the weights of each layer
# for name in layer_names:
#     layer_weights = [model_weights[name][n] for n in range(len(model_weights[name]))]
    # print(layer_weights)

# with h5py.File('cifar10_classifier1.h5', 'r') as f:
#     layer_names = list(f.keys())
#     print(layer_names)
    # w = []
    # b = []

#     for name in layer_names:
#         if 'model_weights' in name or 'optimizer_weights' in name:
#             w =w.append(f[name][name]['kernel:0'][:])
#             print('W111:',w)
#             b.append(f[name][name]['bias:0'][:])
# # Print the number of weight matrices and bias vectors
# # print('Number of weight matrices:', len(w))
# # print('Number of bias vectors:', len(b))
#     # for i in range(len(f.keys())):
#     #     layer_name = 'conv2d_' + str(i)
#     #     layer = f[layer_name]
#     #     w.append(layer['weights'][0])
#     #     b.append(layer['weights'][1])
# print('W', w)
# Define the sigmoid activation function
# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))

# # Perform the forward pass through the model
# def predict(x, w, b):
#     a = x
#     for i in range(len(w)):
#         z = np.dot(a, w[i]) + b[i]
#         a = sigmoid(z)
#     return np.argmax(a, axis=1)

# # Make predictions on the test data
# y_pred = predict(x_test, w, b)

# # Calculate the accuracy of the predictions
# acc = np.mean(y_pred == y_test)
# print('Test accuracy:', acc)