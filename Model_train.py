"""  """

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
from models import VGG_1, VGG_2, VGG_3, define_model_Alexnet, model_New

folder = str(input('Enter the folder address:'))
Tr=int(input('Enter number of Training samples:'))
Val=int(input('Enter number of Validation samples:'))
Ts=int(input('Enter number of Testing samples:'))
e=int(input('Enter number of epochs:'))

a=Tr+Val
b=a+Ts

X = np.load(folder+'Digits_Photo.npy')
Y = np.load(folder+'Digits_labels.npy')
x_train=X[0:Tr]
x_val=X[Tr:a]
x_test=X[a:b]

y=to_categorical(Y)
y_train=y[0:Tr]
y_val=y[Tr:a]
y_test=y[a:b]

# 0:14000, 14000:18000, 18000:20400

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
history=model.fit_generator(train_generator,steps_per_epoch=len(train_generator), epochs=e, validation_data=valid_generator, validation_steps=len(valid_generator),verbose=2)
_, acc = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=0)
print('Test Accuracy= %.3f' % (acc * 100.0))
summarize_diagnostics(history)

