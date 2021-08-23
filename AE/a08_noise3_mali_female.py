import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train =  np.load('./_save/_npy/k59_manwoman_train_x.npy')
x_test =  np.load('./_save/_npy/k59_manwoman_test_x.npy')
x_predic =  np.load('./_save/_npy/k59_manwoman_predic_x.npy')


x_train_noised = x_train + np.random.normal(0,0.1,size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.1,size=x_test.shape)
x_predic_noised = x_predic + np.random.normal(0,0.1,size=x_test.shape)
x_train_noised = np.clip(x_train_noised,a_min=0,a_max=1)
x_test_noised = np.clip(x_test_noised,a_min=0,a_max=1)
x_predic_noised = np.clip(x_test_noised,a_min=0,a_max=1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten

def autoencoder(hidden_layer_size):
     model = Sequential()
     model.add(Dense(units=hidden_layer_size, input_shape=(150,150,3),activation='relu'))
     model.add(Dense(3,activation='softmax'))
     return model
a=154
model = autoencoder(a)
model.compile(optimizer='adam',loss='mse')
model.fit(x_train_noised,x_train,epochs=10)

y_predic = model.predict([x_predic][-1])

from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5),
     (ax11,ax12,ax13,ax14,ax15),
     (ax6,ax7,ax8,ax9,ax10))= \
     plt.subplots(3,1,figsize=(20,7))
# 이미지를 무작위로 5개 고른다
#ran_image = random.sample(range(y_predic.shape[0]),5)

for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
     ax.imshow(x_predic[ran_image[i]].reshape(28,28),cmap='gray')
     if i == 0:
          ax.set_ylabel('INPUT',size=20)
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])

for i,ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
     ax.imshow(x_predic_noised[ran_image[i]].reshape(28,28),cmap='gray')
     if i == 0:
          ax.set_ylabel('NOISED_INPUT',size=20)
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])

for i,ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
     ax.imshow(y_predic[ran_image[i]].reshape(28,28),cmap='gray')
     if i == 0:
          ax.set_ylabel('OUTPUT',size=20)
     ax.grid(False)
     ax.set_xticks([])
     ax.set_yticks([])     

plt.tight_layout()
plt.show()