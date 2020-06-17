# Multiclass-Image-Classifier

Inspired by an episode (S04E04) on the popular comedy TV series, Silicon Valley (HBO), initially I decided to built a desi version of a Binary Image Classifier (similar to the show which built a Hot Dog or Not a Hot Dog) - Vada Pav or Not a Vada Pav. I trained a Convolutional Neural Network (CNN) model on a dataset built from images of Vada Pav and Burger, where the output was binary classification of test data (1-Vada Pav/ 0-Not A Vada Pav). 
Eventually, to learn about Multi-class Image Classification, I decided to abstract on this project. I decided to include 3 classes of images for training the CNN model: Vada Pav, Burger and Sandwiches. And the result entailed correctly labelling the test day as one of the 3 aforementioned categories (Vada Pav, Burger and Sandwiches).

#############################################################################################

Dependancies to be installed before playing around with this: 

$ sudo easy_install pip
$ sudo pip install --upgrade pip
$ pip3 install tensorflow==1.14
$ pip3 install matplotlib
$ pip3 install numpy 
$ pip3 install tqdm
$ pip3 install opencv-python

#############################################################################################

https://www.kaggle.com/brtknr/sushisandwich? : dataset for sandwich images
https://www.kaggle.com/meemr5/vadapav : dataset for vada pav & burger images

(Cleaned the dataset for corrupt image files)

Total images dataset = ~3000

Split that into training/testing as 80%/20% of the total image dataset.

Designed the training data as 40%/40%/20% of respective sandwich/burger/vadapav images. 

Finally had 963 sandwich images, 1004 burger images and 492 vadapav images.
Total size of training dataset = 2459 images
Total size of testing dataset = 608

Used 20% (490 images) of training data for cross-validation achieving 97.96% accuracy 

#############################################################################################