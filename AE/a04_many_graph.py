import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
     model = Sequential()
     model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
     model.add(Dense(784,activation='sigmoid'))
     return model

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_03 = autoencoder(hidden_layer_size=4)
model_04 = autoencoder(hidden_layer_size=8)
model_05 = autoencoder(hidden_layer_size=16)
model_06 = autoencoder(hidden_layer_size=32)
print('============1node===========')

model_01.compile(optimizer='adam',loss='mse')
model_01.fit(x_train,x_train,epochs=10)

print('============2node===========')

model_02.compile(optimizer='adam',loss='mse')
model_02.fit(x_train,x_train,epochs=10)

print('============4node===========')

model_03.compile(optimizer='adam',loss='mse')
model_03.fit(x_train,x_train,epochs=10)

print('============8node===========')

model_04.compile(optimizer='adam',loss='mse')
model_04.fit(x_train,x_train,epochs=10)

print('============16node===========')

model_05.compile(optimizer='adam',loss='mse')
model_05.fit(x_train,x_train,epochs=10)

print('============32node===========')

model_06.compile(optimizer='adam',loss='mse')
model_06.fit(x_train,x_train,epochs=10)

output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_03 = model_03.predict(x_test)
output_04 = model_04.predict(x_test)
output_05 = model_05.predict(x_test)
output_06 = model_06.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, axes =  plt.subplots(7,5,figsize=(15,15))
ran_image = random.sample(range(output_01.shape[0]),5)

outputs = [x_test,output_01,output_02,output_03,output_04,output_05,output_06]
for row_num, row in enumerate(axes):
     for col_num, ax in enumerate(row):
          ax.imshow(outputs[row_num][ran_image[col_num]].reshape(28,28),cmap='gray')
          ax.grid(False)
          ax.set_xticks([])
          ax.set_yticks([])

plt.show()