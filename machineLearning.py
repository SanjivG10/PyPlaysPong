import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dropout, Conv2D,Dense,Flatten, Activation
import numpy as np

from pongMachine import requiredParam


X,y = requiredParam()

y = np.array(y)
X = X.reshape((-1,65,53,1))



model = Sequential()

#
model.add(Conv2D(32,(3,3),input_shape=(65,53,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

#layer2
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

#layer3
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
model.fit(X,y,epochs=50)
model.save('my_model.h5')
