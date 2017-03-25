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
IMG_SIZE_WIDTH = 128 #larger scale can improve accuracy till 350, concluded by a guy
IMG_SIZE_HEIGTH = 128
IMG_CHANNEL =3


def read(path,resize_h=IMG_SIZE_HEIGTH,resize_w=IMG_SIZE_WIDTH,channels=IMG_CHANNEL,denoise_enhance=False):
    img = Image.open(path).convert('RGB').resize((resize_h, resize_w))
    dat = np.asarray(img)
    return dat

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
