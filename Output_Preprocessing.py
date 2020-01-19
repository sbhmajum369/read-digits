''' Storing the digit labels in a numpy format '''

import cv2
import numpy as np
import os
from numpy import save

folder= input('Enter the folder loc for Labels file:')
os.chdir(folder)

labels = list()
with open("Labels.txt","r") as out:
	for value in out:
		temp = value.split('\n')
		labels.append(temp[0])

print('Total length:',len(labels))
n=int(input("length of output:"))

# Replacing '10' by '0'
for i in range(0,n):
	if labels[i]=='10':
		print('Found')
		labels[i]=0
		print(labels[i])
	

save('Digits_labels.npy', labels)
