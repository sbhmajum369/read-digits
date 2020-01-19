
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
import tensorflow as tf

def VGG_1():

    model3 = Sequential()
    model3.add(Conv2D(32, (5, 3), activation='relu', kernel_initializer='he_uniform', data_format='channels_last', padding='same', input_shape=(64, 32, 3)))
    model3.add(Conv2D(32, (5, 3), activation='relu', kernel_initializer='he_uniform', strides=2, padding='same'))
    model3.add(tf.keras.layers.BatchNormalization())
    model3.add(AveragePooling2D((3, 3)))
    model3.add(Dropout(0.2))
    model3.add(Conv2D(64, (3, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model3.add(Conv2D(64, (3, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model3.add(tf.keras.layers.BatchNormalization())
    model3.add(MaxPooling2D((2, 2)))
    model3.add(Dropout(0.2))
    model3.add(Conv2D(128, kernel_size=(2, 2), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model3.add(Conv2D(128, kernel_size=1, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model3.add(Flatten())
    model3.add(Dense(128, activation='tanh', kernel_initializer='he_uniform'))
    model3.add(Dropout(0.2))
    model3.add(Dense(64, activation='tanh', kernel_initializer='he_uniform'))
    model3.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model3.add(Dense(10, activation='softmax'))
    model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model3

def VGG_2():

    model4 = Sequential()
    model4.add(Conv2D(32, (5, 3), activation='relu', kernel_initializer='he_uniform', data_format='channels_last', strides=1, padding='same', input_shape=(64, 32, 3)))
    model4.add(Conv2D(32, (5, 3), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model4.add(AveragePooling2D((2, 2)))
    model4.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model4.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
    model4.add(MaxPooling2D((2, 2)))
    # model4.add(Dropout(0.2))
    model4.add(Conv2D(128, kernel_size=1, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model4.add(Dropout(0.2))
    model4.add(MaxPooling2D((2, 2)))
    model4.add(Flatten())
    model4.add(Dense(150, activation='relu', kernel_initializer='he_uniform'))
    # model4.add(Dropout(0.2))
    model4.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model4.add(Dropout(0.2))
    model4.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model4.add(Dense(10, activation='softmax'))
    model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model4



def VGG_3():

  model5 = Sequential()
  model5.add(Conv2D(32, (7, 5), activation='relu', data_format='channels_last', kernel_initializer='he_uniform', padding='same', input_shape=(64, 32, 3)))
  model5.add(Conv2D(64, (7, 5), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
  model5.add(MaxPooling2D((2, 2)))
  model5.add(Dropout(0.2))
  model5.add(Conv2D(64, (5, 3), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
  model5.add(Conv2D(128, (5, 3), activation='relu', kernel_initializer='he_uniform', strides=1, padding='same'))
  model5.add(MaxPooling2D((2, 2)))
  model5.add(Dropout(0.2))
  model5.add(Conv2D(128, (3, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model5.add(Conv2D(128, (3, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model5.add(MaxPooling2D((3, 3)))
  model5.add(Dropout(0.2))
  model5.add(Flatten())
  model5.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model5.add(Dropout(0.2))
  model5.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
  model5.add(Dense(10, activation='softmax'))
  model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model5

def model_Alexnet():
  model1 = Sequential()
  model1.add(Conv2D(96, (9, 7), activation='relu', data_format='channels_last', kernel_initializer='he_uniform', padding='same', input_shape=(64, 32, 3)))
  model1.add(MaxPooling2D((3, 3)))
  model1.add(Conv2D(256, (7, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model1.add(MaxPooling2D((3, 3)))
  model1.add(Dropout(0.2))
  model1.add(Conv2D(384, (5, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model1.add(Conv2D(256, (5, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model1.add(MaxPooling2D((3, 3)))
  model1.add(Dropout(0.2))
  model1.add(Flatten())
  model1.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model1.add(Dropout(0.2))
  model1.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
  model1.add(Dense(10, activation='softmax'))
  model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model1

def model_New():
  model2 = Sequential()
  model2.add(Conv2D(48, (5, 5), activation='relu', data_format='channels_last', kernel_initializer='he_uniform', padding='same', input_shape=(64, 32, 3)))
  model2.add(MaxPooling2D((3, 3)))
  model2.add(Dropout(0.2))
  model2.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model2.add(MaxPooling2D((2, 2)))
  model2.add(Dropout(0.2))
  model2.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model2.add(MaxPooling2D((2, 2)))
  model2.add(Dropout(0.2))
  model2.add(Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model2.add(Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model2.add(MaxPooling2D((2, 2)))
  model2.add(Dropout(0.2))
  model2.add(Conv2D(192, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model2.add(Conv2D(192, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model2.add(Dropout(0.2))
  model2.add(Flatten())
  model2.add(Dense(180, activation='relu', kernel_initializer='he_uniform'))
  model2.add(Dropout(0.2))
  model2.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
  model2.add(Dropout(0.2))
  model2.add(Dense(60, activation='relu', kernel_initializer='he_uniform'))
  model2.add(Dense(10, activation='softmax'))
  model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model2

