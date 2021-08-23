# 함수를 이용해서 딥하게 구성하여 비교분석(CNN)

import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()


from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input,Conv2D,MaxPooling2D,UpSampling2D,Flatten

x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255 #loss: 7215.1093

x_train2 = x_train.reshape(60000, 28*28, 1).astype('float')/255

x_test = x_test.reshape(10000, 28, 28, 1).astype('float')/255
def autoencoder(hidden_layer_size):
     model = Sequential()
     model.add(Conv2D(filters=hidden_layer_size,kernel_size=(2,2), input_shape=( 28, 28, 1),activation='relu',padding='same'))
     model.add(MaxPooling2D())
     model.add(Conv2D(16,(2,2),activation='relu',padding='same'))
     model.add(MaxPooling2D())
     model.add(Conv2D(16,(2,2),activation='relu',padding='same'))     
     model.add(Flatten())
     model.add(Dense(784,activation='sigmoid'))
     return model

def autodecoder(hidden_layer_size):
     model = Sequential()
     model.add(Conv2D(filters=hidden_layer_size,kernel_size=(2,2), input_shape=( 28, 28, 1),activation='relu',padding='same'))
     model.add(UpSampling2D(size=(2,2)))
     model.add(Conv2D(16,(2,2),activation='relu',padding='same'))
     model.add(MaxPooling2D())
     model.add(Conv2D(16,(2,2),activation='relu',padding='same'))
     model.add(MaxPooling2D())
     model.add(Conv2D(16,(2,2),activation='relu',padding='same'))    
     model.add(Flatten())
     model.add(Dense(784,activation='sigmoid'))
     return model

model_01 = autoencoder(hidden_layer_size=128)
model_02 = autodecoder(hidden_layer_size=784)

model_01.compile(optimizer='adam',loss='mse')
model_01.fit(x_train,x_train2,epochs=10)

model_02.compile(optimizer='adam',loss='mse')
model_02.fit(x_train,x_train2,epochs=10)

output_en = model_01.predict(x_test)
output_de = model_02.predict(x_test)



from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),
     (ax11,ax12,ax13,ax14,ax15),
     (ax6,ax7,ax8,ax9,ax10)) = \
     plt.subplots(3,5,figsize=(20,7))
# 이미지를 무작위로 5개 고른다
ran_image = random.sample(range(output_en.shape[0]),5)
ran_image2 = random.sample(range(output_de.shape[0]),5)

for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
     ax.imshow(x_train[ran_image[i]].reshape(28,28),cmap='gray')
     if i == 0:
          ax.set_ylabel('INPUT',size=20)
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])

for i,ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
     ax.imshow(output_en[ran_image[i]].reshape(28,28),cmap='gray')
     if i == 0:
          ax.set_ylabel('en',size=20)
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])     

for i,ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
     ax.imshow(output_de[ran_image2[i]].reshape(28,28),cmap='gray')
     if i == 0:
          ax.set_ylabel('de',size=20)
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])
plt.tight_layout()
plt.show()
'''
from matplotlib import pyplot as plt
import random

fig, axes =  plt.subplots(7,5,figsize=(15,15))
ran_image = random.sample(range(model_01.shape[0]),5)

outputs = [x_test,model_01,model_02]
for row_num, row in enumerate(axes):
     for col_num, ax in enumerate(row):
          ax.imshow(outputs[row_num][ran_image[col_num]].reshape(28,28),cmap='gray')
          ax.grid(False)
          ax.set_xticks([])
          ax.set_yticks([])

plt.show()
'''