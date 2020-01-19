''' Resizing the images and Storing them as numpy array '''

import cv2
import numpy as np
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os

os.chdir('D:/Fall 2019/Machine and Deep Learning/Project/New _Dataset/Train/')	# Home directory of images

n=int(input("Enter num of images for Training and Val:"))

folder = 'D:/Fall 2019/Machine and Deep Learning/Project/New _Dataset/Testing'	# New directory for storing resized images

for img in range(1,n+1):
	img_data=cv2.imread(str(img)+'.png')
	new_img=cv2.resize(img_data, (32,64), interpolation = cv2.INTER_AREA)
	cv2.imwrite(os.path.join(folder , str(img)+'.png'), new_img)

# print("....Resizing Complete....")
os.chdir(folder)

photos= list()
# Enumerate files in the directory
for file in range(1,n+1):
	print(file)
	photo = cv2.imread(str(file)+'.png')
	# photo = load_img(str(file)+'.png')  		# Can Add: target_size=(200, 200)
	photo = img_to_array(photo)					# Convert to numpy array
	photos.append(photo)
	

photos = asarray(photos)					# Storing into a singular array file
print(photos.shape)
save('Digits_Photo.npy', photos)


