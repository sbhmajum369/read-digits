"""  """

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
from models import VGG_1, VGG_2, VGG_3, define_model_Alexnet, model_New

X = np.load('Digits_Photo.npy')
Y = np.load('Digits_labels.npy')
x_train=X[0:14000]
x_val=X[14000:18000]
x_test=X[18000:20400]

y=to_categorical(Y)
y_train=y[0:14000]
y_val=y[14000:18000]
y_test=y[18000:20400]

datagen = ImageDataGenerator()		# rescale=1.0/255.0  rotation_range=20
train_generator=datagen.flow(x_train, y_train, batch_size=100)
valid_generator=datagen.flow(x_val, y_val, batch_size=50)
test_generator=datagen.flow(x_test,y_test, batch_size=50)

def summarize_diagnostics(history):
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	plt.show()
	plt.title('Classification Accuracy')
	plt.plot(history.history['acc'], color='blue', label='train')
	plt.plot(history.history['val_acc'], color='orange', label='test')
	plt.show()

model = VGG_1()
history=model.fit_generator(train_generator,steps_per_epoch=len(train_generator), epochs=30, validation_data=valid_generator, validation_steps=len(valid_generator),verbose=2)
_, acc = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=0)
print('Test Accuracy= %.3f' % (acc * 100.0))
summarize_diagnostics(history)

