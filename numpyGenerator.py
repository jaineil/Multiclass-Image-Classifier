# Generates a Numpy array of the images in the dataset. It was (X, 2) dimensions. 
# X -> number of images in the dataset as rows
# 2 columns -> grayscale image & label for it

import os
import cv2                 
import numpy as np
from tqdm import tqdm


TRAIN1 = '/Users/jaineilmandavia/Desktop/train-data/sandwich'
TRAIN2 = '/Users/jaineilmandavia/Desktop/train-data/burger'
TRAIN3 = '/Users/jaineilmandavia/Desktop/train-data/vadapav'
TEST = '/Users/jaineilmandavia/Desktop/test-data'

IMG_SIZE = 50

###############################################################################

def label_image(name):
    word_label = name
    
    if word_label == 'sandwich': return [1,0,0]
    elif word_label == 'burger': return [0,1,0]
    else: return [0,0,1]

def create_train_data():
    training_data = []
    
    for img in tqdm(os.listdir(TRAIN1)):
        label = label_image("sandwich")
        try:
            path = os.path.join(TRAIN1,img)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            #cv2.imshow('image', img)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        except Exception as e:
            print(str(e))
        training_data.append([np.array(img),np.array(label)])

    for img in tqdm(os.listdir(TRAIN2)):
        label = label_image("burger")
        try:
            path = os.path.join(TRAIN2,img)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            #cv2.imshow('image', img)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        except Exception as e:
            print(str(e))
        training_data.append([np.array(img),np.array(label)])

    for img in tqdm(os.listdir(TRAIN3)):
        label = label_image("vadapav")
        try:
            path = os.path.join(TRAIN3,img)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            #cv2.imshow('image', img)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        except Exception as e:
            print(str(e))
        training_data.append([np.array(img),np.array(label)])

    np.save('train_data.npy', training_data)
    return training_data

###############################################################################

train_data = create_train_data()

###############################################################################

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST)):
        img_num = img.split('.')[0]
        try:
            path = os.path.join(TEST,img)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            #print(type(img))
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        except Exception as e:
            print(str(e)+img_num)
            #print(img_num)
        testing_data.append([np.array(img), img_num])
    
    #shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

###############################################################################

test_data = process_test_data()

###############################################################################
