# Reading-digits
Reading alphanumeric digits from real-world images is a very hard task to solve. While OCR techniques combined with other algorithms have successfully achieved that objective for binary documents and images, the task still remains for the images in real-world scenerio. There, different conditions such as brightness, color and other backgrounds affect the identification and extraction of numbers and characters.

Deep Learning can, however, solve these issues by taking each pixels into account. Using CNNs, the task of character recognition in natural-scene images can be solved very efficiently.

The following deep learning model was designed for classifying the individual characters after extracting them from the full image. For this, the Street View House Number(SVHN) dataset has been used. It has a large amount of collection of these natural images containing house numbers, which were collected from Google Street View images.

Link: http://ufldl.stanford.edu/housenumbers/

A MATLAB code is provided alongside the dataset for extracting individual digits, along with their labels.

## Steps for Training and Testing

Clone the repo using: git clone https://github.com/smajum-AI/read-digits.git


A) Install all the dependencies if you are using an offline environment, such as PC or Mac. You can create a virtual environment in your project folder and install all the dependencies that doesn't come with python installation.

For this you will need:
1) TensorFlow
2) OpenCV
3) Matplotlib

B) After installation, prepare your dataset into an 'Image' folder and a 'Label.txt' file.

Here, the SVHN dataset has been used.
For that, run the see_bboxes.m file in MATLAB and provide the appropriate folder location with the above names.

C) After dataset preparation, place the image folder and the label file within the project folder and use their location to run the pre-processing files: 'Image_Preprocess.py' and 'Output_Preprocessing.py'. 

Now we will have 2 numpy files: One storing the images and another their corresponding labels, for training, validation and testing.

D) Run the 'Model_train.py' file for training and testing, through the command line and provide the following data sequentially:

1) location of the numpy files
2) Training, validation and test sizes
3) Number of epochs. 

On Line 46 of 'Model_train.py', select which model to train with from the 'models.py' file.

Both the VGG models, provided 97% of accuracy during testing, using a total dataset size of 40,000 images.
