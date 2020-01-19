# Reading-digits
Reading alphanumeric digits from real-world images is a very hard task to solve. While OCR techniques combined with other algorithms have successfully achieved that objective for binary documents and images, the task still remains for the images in real-world scenerio. There, different conditions such as brightness, color and other backgrounds affect the identification and extraction of numbers and characters.

Deep Learning can, however, solve these issues by taking each pixels into account. Using CNNs the task of character recognition in natural-scene images can be solved very efficiently.

The Street View House Number(SVHN) dataset has a large amount of collection of these natural images collected from Google Street View images.
Link: http://ufldl.stanford.edu/housenumbers/
This code is designed for Format 1 dataset.

A MATLAB code is provided for extracting individual digits, along with their labels, from the images using the matrix file.

After extraction, run the preprocessing files. In 'Model_train', choose training, validation and test sizes in a uniform way.

