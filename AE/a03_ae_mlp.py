# 함수를 이용해서 딥하게 구성하여 비교분석

import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder1(hidden_layer_size):
     model = Sequential()
     model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
     model.add(Dense(784,activation='sigmoid'))
     return model

def autoencoder2(hidden_layer_size):
     model = Sequential()
     model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
     model.add(Dense(1000))
     model.add(Dense(784,activation='sigmoid'))
     return model

a=154
model = autoencoder2(a)
model.compile(optimizer='adam',loss='mse')
model.fit(x_train,x_train,epochs=10)
output = model.predict(x_test)
from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10)) = \
     plt.subplots(2,5,figsize=(20,7))
# 이미지를 무작위로 5개 고른다
ran_image = random.sample(range(output.shape[0]),5)

for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
     ax.imshow(x_test[ran_image[i]].reshape(28,28),cmap='gray')#cmp='gray'
     if i == 0:
          ax.set_ylabel('INPUT',size=20)
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])

for i,ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
     ax.imshow(output[ran_image[i]].reshape(28,28),cmap='gray')
     if i == 0:
          ax.set_ylabel('OUTPUT',size=20)
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])     

plt.tight_layout()
plt.show()