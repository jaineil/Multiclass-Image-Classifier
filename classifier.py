import cv2                 
import numpy as np         
import os
import matplotlib.pyplot as plt
from tqdm import tqdm      
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 1e-3
MODEL_NAME = 'sandwich_burger_vadapav-{}-{}.model'.format(LR, '6conv-basic') 
IMG_SIZE =  50 

# First create Numpy arrays of training data and testing data using the 'numpyGenerator.py' file
# If you have already created a Numpy arrays of the dataset using the 'numpyGenerator.py' 
# Simply 'import' those file
# Creating dataset as Numpy array locally allows one to skip the process of.. 
# ..Adding all the data to Google Drive and Mounting it on Google Colab

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

###############################################################################

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

###############################################################################

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded!')

############################################################################### 

train = train_data[:-490]
validation_test = train_data[-490:]

X = np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train_data]

test_x = np.array([i[0] for i in validation_test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in validation_test]

model.fit({'input': X}, {'targets': Y}, n_epoch=25, 
    snapshot_step=500, show_metric=True, validation_set=({'input': test_x}, {'targets': test_y}), run_id=MODEL_NAME)
model.save(MODEL_NAME)   

###############################################################################

fig=plt.figure()

for num, data in enumerate(test_data[:72]):
    
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(8,9,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 0: str_label='Sandwich'
    elif np.argmax(model_out) == 1: str_label='Burger'
    else: str_label='VadaPav'        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.axis('off')
plt.show()