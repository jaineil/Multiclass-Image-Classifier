import cv2                 
import numpy as np         
import os
import matplotlib.pyplot as plt
from random import shuffle 
from tqdm import tqdm      
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 1e-3
MODEL_NAME = 'sandwich_burger_vadapav-{}-{}.model'.format(LR, '6conv-basic') 
IMG_SIZE =  50

''' 
TRAIN1 = '/Users/jaineilmandavia/GitHub/Multiclass-Image-Classifier/train-data/sandwich'
TRAIN2 = '/Users/jaineilmandavia/GitHub/Multiclass-Image-Classifier/train-data/burger'
TRAIN3 = '/Users/jaineilmandavia/GitHub/Multiclass-Image-Classifier/train-data/vadapav'
TEST = '/Users/jaineilmandavia/GitHub/Multiclass-Image-Classifier/test-data'

# IMG_SIZE = 50
# LR = 1e-3

# MODEL_NAME = 'vadapav_vs_notvadapav-{}-{}.model'.format(LR, '6conv-basic') 

def label_image(name):
    word_label = name
    
    if word_label == 'sandwich': return [1,0,0]
    elif word_label == 'burger': return [0,1,0]
    else: return [0,0,1]

def create_train_data():
    training_data = []
    
    for img in tqdm(os.listdir(TRAIN1)):
        label = label_image("sandwich")
        path = os.path.join(TRAIN1,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('image', img)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    #shuffle(training_data)

    for img in tqdm(os.listdir(TRAIN2)):
        label = label_image("burger")
        path = os.path.join(TRAIN2,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('image', img)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    #shuffle(training_data)

    for img in tqdm(os.listdir(TRAIN3)):
        label = label_image("vadapav")
        path = os.path.join(TRAIN3,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #cv2.imshow('image', img)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    #shuffle(training_data)

    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST)):
        path = os.path.join(TEST,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #print(type(img))
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    
    #shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
test_data = process_test_data()
'''

# If you have already created a Numpy array of the dataset using the 'numpyGenerator.py' 
# And want to run this on cloud (Colab):
# Simply 'import' the train_data.npy and test_data.npy on Colab!

train_data = np.load('train_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)

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

#train_data.shape = (1860, 2) => 1860 image rows, 2 columns (grayscale image, label)

train = train_data[:-336]
validation_test = train_data[-336:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=30, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME) 

###############################################################################

fig=plt.figure()

for num, data in enumerate(test_data):
    
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(8,9,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 0: str_label='Sandwich'
    elif np.argmax(model_out) == 1: str_lab='Burger'
    else: str_label='VadaPav'        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.axis('off')
plt.show()

# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# for num, data in enumerate(test_data):
#     img_data = data[0]
#     data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
#     model_out = model.predict([data])[0]
#     if np.argmax(model_out) == 0: str_label='Sandwich'
#     elif np.argmax(model_out) == 1: str_lab='Burger'
#     else: str_label='VadaPav' 
#     print(str_label)
#     #img = cv2.imread(TEST)
#     #cv2.imshow('image', img)
#     path = '/Users/jaineilmandavia/GitHub/Multiclass-Image-Classifier/test-data'
#     files = os.listdir(path)
#     #print(files)
#     for file in files:
#         os.rename(os.path.join(path, file), os.path.join(path, str_label + '.jpg'))   
#     img = cv2.imread('/Users/jaineilmandavia/GitHub/Multiclass-Image-Classifier/test-data/%s.jpg' %str_label)
#     img1 = cv2.resize(img, (768,1024))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     if str_label == "Vadapav!":
#         cv2.rectangle(img1, (0,0),(768,100),(0,255,0),-1)
#         cv2.circle(img1, (384, 100), 60, (0,255,0),-1)
#         cv2.putText(img1, str_label, (250,70), font, 2, (255,255,255),4,cv2.LINE_AA)
#     else:
#         cv2.rectangle(img1, (0,0),(768,100),(0,0,255),-1)
#         cv2.circle(img1, (384, 100), 60, (0,0,255),-1)
#         cv2.putText(img1, str_label, (130,70), font, 2, (255,255,255),4,cv2.LINE_AA)
#     cv2.imshow("SeeFood", img1)
#     cv2.waitKey()
#     cv2.destroyAllWindows()