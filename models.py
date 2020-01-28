
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
import tensorflow as tf

def VGG_1():  # 97.8 %
    model1 = Sequential()
    model1.add(Conv2D(32, (5, 3), activation='relu', kernel_initializer='he_uniform', data_format='channels_last', padding='same', input_shape=(64, 32, 3)))
    model1.add(Conv2D(32, (5, 3), activation='relu', kernel_initializer='he_uniform', strides=2, padding='same'))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(AveragePooling2D((3, 3)))
    model1.add(Dropout(0.2))
    model1.add(Conv2D(64, (3, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model1.add(Conv2D(64, (3, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model1.add(tf.keras.layers.BatchNormalization())
    model1.add(MaxPooling2D((2, 2)))
    model1.add(Dropout(0.2))
    model1.add(Conv2D(128, kernel_size=(2, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model1.add(Conv2D(128, kernel_size=1, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model1.add(Flatten())
    model1.add(Dense(128, activation='tanh', kernel_initializer='he_uniform'))
    model1.add(Dropout(0.2))
    model1.add(Dense(64, activation='tanh', kernel_initializer='he_uniform'))
    model1.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model1.add(Dense(10, activation='softmax'))
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model1


def VGG_2():    # 97.12 %
    model2 = Sequential()
    model2.add(Conv2D(32, (4, 2), activation='relu', kernel_initializer='he_uniform', data_format='channels_last', strides=1, padding='same', input_shape=(64, 32, 3)))
    model2.add(Conv2D(32, (4, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model2.add(AveragePooling2D((2, 2)))
    model2.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model2.add(Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model2.add(AveragePooling2D((2, 2)))
    model2.add(Conv2D(128, kernel_size=1, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dense(150, activation='relu', kernel_initializer='he_uniform'))
    model2.add(Dropout(0.2))
    model2.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model2.add(Dropout(0.2))
    model2.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model2.add(Dense(10, activation='sigmoid'))
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model2

