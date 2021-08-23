# 앞 뒤가 똑같은 오~토인코더~~~
# 약한 특성의 소멸

import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
# encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(1064, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoded = Model(input_img,decoded)
autoencoded.summary()

'''
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
dense (Dense)                (None, 64)                50240
_________________________________________________________________
dense_1 (Dense)              (None, 784)               50960
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________

'''

autoencoded.compile(optimizer='adam',loss='binary_crossentropy')
autoencoded.fit(x_train,x_train,epochs=30,batch_size=128,validation_split=0.2)
#x가 들어가서 다시 x가 나오기때문에 y는 필요없다
decode_image = autoencoded.predict(x_test)

import matplotlib.pyplot as plt

n = 20
plt.figure(figsize=(20,4))

for i in range(n):
     ax = plt.subplot(2,n,i+1)
     plt.imshow(x_test[i].reshape(28,28))
     plt.gray()
     ax.get_xaxis().set_visible(False)
     ax.get_yaxis().set_visible(False)
     
     ax = plt.subplot(2,n,i+1)
     plt.imshow(decode_image[i].reshape(28,28))
     plt.gray()
     ax.get_xaxis().set_visible(False)
     ax.get_yaxis().set_visible(False)

plt.show()