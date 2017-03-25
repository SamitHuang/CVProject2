#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Functions for downloading and reading MNIST data."""
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
from os.path import join
import tempfile
import random
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import maybe_download
import GetInputData as gid
import convert_to_records
import numpy as np
import ImageProcessor

from PIL import Image, ImageEnhance, ImageFilter

cwd = os.getcwd()

IMG_HEIGHT = convert_to_records.IMG_HEIGHT
IMG_WIDTH = convert_to_records.IMG_WIDTH
IMG_CHANNELS = convert_to_records.IMG_CHANNELS
IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

TRAIN_DATA_DIR='data_small/train/'
TEST_DATA_DIR='data_small/test/'
TFR_SAVE_DIR='data_small/'
NUM_TRAIN =2000
NUM_VALI = 1000

IMG_PIXELS2 = 350*350*3;




def get_train_and_vali_data(path=TRAIN_DATA_DIR,num_train=NUM_TRAIN,num_vali=NUM_VALI):
    '''
    :return: train data, numpy array, 每一个元素包含一个图片数据(np格式)和对应的lable
             test data,
    '''
    #cat file name
    NUM_TRAIN_DOGS = num_train / 2
    NUM_TRAIN_CATS = num_train / 2
    NUM_VALI_DOGS = num_vali / 2
    NUM_VALI_CATS = num_vali / 2
    trainDogImgs=[]
    trainCatImgs=[]
    valiDogImgs=[]
    valiCatImgs=[]
    trainX=[]
    trainY=[]
    valiX=[]
    valiY=[]
    #trainX = np.zeros((num_train, ImageProcessor.IMAGE_SIZE_HEIGTH, ImageProcessor.IMAGE_SIZE_WIDTH, img_channels), dtype=np.uint8)
    #trainY = np.zeros((num_train,), dtype=np.uint8)

    for (i,imgName) in enumerate(os.listdir(path)):
        if ('dog' in imgName):
            if (len(trainDogImgs) < NUM_TRAIN_DOGS):
                trainDogImgs.append([ imgName,1])
            elif (len(valiDogImgs) < NUM_VALI_DOGS):
                valiDogImgs.append([imgName,1])
        if('cat' in imgName):
            if (len(trainCatImgs) < NUM_TRAIN_CATS):
                trainCatImgs.append([ imgName,0])
            elif (len(valiCatImgs) < NUM_VALI_CATS):
                valiCatImgs.append([ imgName,0])
        if(len(valiCatImgs)+len(valiDogImgs) ==NUM_VALI):
            break;

    trainImgs=trainDogImgs + trainCatImgs;
    valiImgs = valiDogImgs + valiCatImgs;

    random.shuffle(trainImgs)

    for i,img_name_label in enumerate(trainImgs):
        dat=ImageProcessor.read(TRAIN_DATA_DIR + img_name_label[0])
        #trainData.append([dat,imgPath[1]])
        trainX.append(dat)
        #trainX[i, :, :, :]=dat
        trainY.append(img_name_label[1])
    for i,img_name_label in enumerate(valiImgs):
        dat = ImageProcessor.read(TRAIN_DATA_DIR + img_name_label[0])
        valiX.append(dat)
        #valiX[i,:,:,:]=dat
        valiY.append(img_name_label[1])

    #return np.array(trainData),np.array(testData)
    return np.array(trainX),np.array(trainY),np.array(valiX),np.array(valiY)


def convert_to_tfr(images, labels, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = join(TFR_SAVE_DIR, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        label = labels[index]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),   # NOT assuming one-hot format of original data
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_save_data():
    train_images,train_labels,vali_images,vali_labels=get_train_and_vali_data()
    #print(train_images.shape)
    #convert_to_records.convert_to(train_images,train_lables,"train")

    convert_to_tfr(train_images,train_labels,"train")
    convert_to_tfr(vali_images,vali_labels,"validation")


'''
def save_data_to_tfrecords()
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((224, 224))
            img_raw = img.tobytes()              #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())  #序列化为字符串


    writer.close()
'''



def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    img_height = tf.cast(features['height'], tf.int32)
    img_width = tf.cast(features['width'], tf.int32)
    img_depth = tf.cast(features['depth'], tf.int32)
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    image.set_shape([IMG_PIXELS])
    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    return image, label



def read_and_decode2(filename):
    #params: tfrecords文件名
    #使用tensorflow队列高速读取文件,是符号类型，在sess.run()时才启动
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                            'image_raw': tf.FixedLenFeature([], tf.string),
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'height': tf.FixedLenFeature([], tf.int64),
                                            'width': tf.FixedLenFeature([], tf.int64),
                                            'depth': tf.FixedLenFeature([], tf.int64)
                                       })

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img_height = tf.cast(features['height'], tf.int32)
    img_width = tf.cast(features['width'], tf.int32)
    img_depth = tf.cast(features['depth'], tf.int32)
    #img = tf.reshape(img, [224, 224, 3])


    img.set_shape(IMG_PIXELS)
    img = tf.reshape(img, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

def test_read_tfrecords():
    # 启动tf.Session后，tfrecords将不断pop出文件名
    tfr_fn = 'data_small/train.tfrecords'
    img, label = read_and_decode2(tfr_fn)
    # 使用shuffle_batch可以随机打乱输入
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=8, capacity=2000,
                                                    min_after_dequeue=1000)
    # min_after_dequeue,出队列后，队列最小有多长。与train数据、batch_size有关。如果有tfr train有1000张dog，1000张cat，只各取500张出来训
    # bacth_size，一次出8张，为随机的一个集合
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # threads = tf.train.start_queue_runners(sess=sess)
        try:
            # for i in range(3):
            step = 0
            step_wanted = 100
            while (not coord.should_stop()) and (step < step_wanted):
                step += 1
                val, l = sess.run([img_batch, label_batch])
                # 我们也可以根据需要对val， l进行处理
                # l = to_categorical(l, 12)

                if ((1 in l) and (0 in l)):
                    print(val.shape, l)
                    for j in range(3):
                        Image.fromarray(((val[j] + 0.5) * 255).astype(np.uint8)).show()
        except tf.errors.OutOfRangeError:
            print('Done pop out all data in the tfrecords')
        finally:
            coord.request_stop()
            print("DEBUG: try to finally")
        coord.join(threads)

if __name__ == "__main__":
    #read_and_save_data()
    test_read_tfrecords()
    '''
    data_sets=read_data_sets()
    images_feed, labels_feed = data_sets.next_batch(4)
    print(labels_feed)
    '''