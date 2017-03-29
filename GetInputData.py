#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import random
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf

#data setting
TRAIN_DATA_DIR='../data/train/'
NUM_TRAIN = 20000
NUM_TEST = 600
NUM_TRAIN_DOGS = NUM_TRAIN/2
NUM_TRAIN_CATS = NUM_TRAIN/2
NUM_TEST_DOGS=NUM_TEST/2
NUM_TEST_CATS=NUM_TEST/2

IMAGE_SIZE_WIDTH = 128 #larger scale can improve accuracy till 350, concluded by a guy
IMAGE_SIZE_HEIGTH = 128

TEST_DATA_DIR='../data/test/'
IS_LIMIT_NUM_TEST_UNKNOWN=False
NUM_TEST_UNKNOWN = 111 #12500

def GetImageLable(imgName):
    '''
    :param imgDir:
    :return: 1 if dog, 0 if cat
    '''
    if('dog' in imgName):
        return 1
    if('cat' in imgName):
        return 0

def ProcessImage(imgPath):
    '''
    read one image and reshape it (or enhance)
    :param imgPath:
    :return:
    '''
    #img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    #img = cv2.resize(img,(IMAGE_SIZE_HEIGTH,IMAGE_SIZE_WIDTH))
    img = Image.open(imgPath).convert('RGB').resize((IMAGE_SIZE_HEIGTH, IMAGE_SIZE_WIDTH))
    dat = np.asarray(img)
    return dat

def GetTrainAndValidateData():
    '''
    :return: train data, numpy array, 每一个元素包含一个图片数据(np格式)和对应的lable
             test data,
    '''
    #cat file name
    trainDogImgs=[]
    trainCatImgs=[]
    testDogImgs=[]
    testCatImgs=[]
    trainX=[]
    trainY=[]
    testX=[]
    testY=[]

    for (i,imgName) in enumerate(os.listdir(TRAIN_DATA_DIR)):
        if ('dog' in imgName):
            if (len(trainDogImgs) < NUM_TRAIN_DOGS):
                trainDogImgs.append([ imgName,1])
            elif (len(testDogImgs) < NUM_TEST_DOGS):
                testDogImgs.append([imgName,1])
        if('cat' in imgName):
            if (len(trainCatImgs) < NUM_TRAIN_CATS):
                trainCatImgs.append([ imgName,0])
            elif (len(testCatImgs) < NUM_TEST_CATS):
                testCatImgs.append([ imgName,0])
        if(len(testCatImgs)+len(testDogImgs) ==NUM_TEST):
            break;

    trainImgs=trainDogImgs + trainCatImgs;
    testImgs = testDogImgs + testCatImgs;

    random.shuffle(trainImgs)

    for imgPath in trainImgs:
        dat=ProcessImage(TRAIN_DATA_DIR + imgPath[0])
        #trainData.append([dat,imgPath[1]])
        trainX.append(dat)
        trainY.append(imgPath[1])
    for imgPath in testImgs:
        dat = ProcessImage(TRAIN_DATA_DIR + imgPath[0])
        #testData.append([dat, imgPath[1]])
        testX.append(dat)
        testY.append(imgPath[1])

    #return np.array(trainData),np.array(testData)
    return np.array(trainX),np.array(trainY),np.array(testX),np.array(testY)

'''
def GetTestData():
    #for (i, imgName) in enumerate(os.listdir(TRAIN_DATA_DIR)):
    img_names =os.listdir(TEST_DATA_DIR)[:NUM_TEST] # [TEST_DATA_DIR + i for i in ]
    test_data=[]
    img_id=[]
    for imgn in img_names:
        dat = ProcessImage(TEST_DATA_DIR + imgn)
        test_data.append(dat)
        img_id.append(imgn.split('.')[0])
    np.save('test_data.npy', test_data)
    np.save('test_data_id.npy', img_id)
    return np.array(test_data,dtype=np.float32),img_id
'''
def GetTestData():
    test_data=[]
    img_ids = []
    for imgn in os.listdir(TEST_DATA_DIR):
        img_id = imgn.split('.')[0]
        img_ids.append(int(img_id))
    img_ids.sort()
    if(IS_LIMIT_NUM_TEST_UNKNOWN):
        img_ids=img_ids[0:NUM_TEST_UNKNOWN]
    for img_id in img_ids:
        dat = ProcessImage(TEST_DATA_DIR + str(img_id) + '.jpg')
        test_data.append(dat)
        #img_id.append(imgn.split('.')[0])
    np.save('test_data.npy', test_data)
    np.save('test_data_id.npy', img_ids)
    return np.array(test_data,dtype=np.float32),img_ids



if __name__ == "__main__":
   GetTestData()
