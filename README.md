# Multiclass-Image-Classifier
Inspired by an episode (S04E04) on the popular comedy TV series, Silicon Valley (HBO), initially I decided to built a desi version of a Binary Image Classifier (similar to the show which built a Hot Dog or Not a Hot Dog) - Vada Pav or Not a Vada Pav. I trained a Convolutional Neural Network (CNN) model on a dataset built from images of Vada Pav and Burger, where the output was binary classification of test data (1-Vada Pav/ 0-Not A Vada Pav). 
Eventually, to learn about Multi-class Image Classification, I decided to abstract on this project. I decided to include 3 classes of images for training the CNN model: Vada Pav, Burger and Sandwiches. And the result entailed correctly labelling the test day as one of the 3 aforementioned categories (Vada Pav, Burger and Sandwiches).

####################################################################################################

Dependancies to be installed before playing around with this: 

$ sudo easy_install pip
$ sudo pip install --upgrade pip
$ pip3 install tensorflow==1.14
$ pip3 install matplotlib
$ pip3 install numpy 
$ pip3 install tqdm
$ pip3 install opencv-python

####################################################################################################