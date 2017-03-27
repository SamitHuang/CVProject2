#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf

#data setting
TRAIN_DATA_DIR='../data/train/'
NUM_TRAIN =320
NUM_TEST = 32
NUM_TRAIN_DOGS = NUM_TRAIN/2
NUM_TRAIN_CATS = NUM_TRAIN/2
NUM_TEST_DOGS=NUM_TEST/2
NUM_TEST_CATS=NUM_TEST/2

IMAGE_SIZE_WIDTH = 64 #larger scale can improve accuracy till 350, concluded by a guy
IMAGE_SIZE_HEIGTH = 64

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
