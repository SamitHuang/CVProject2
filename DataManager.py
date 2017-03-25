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


#数据读取的相关参数,
DEFAULT_EPOCH_NUM_LIMIT = None  #None无限制
DEFAULT_EPOCH_SHUFFLE=True  #每换一次epoch shuffle一次


cwd = os.getcwd()

IMG_HEIGHT = convert_to_records.IMG_HEIGHT
IMG_WIDTH = convert_to_records.IMG_WIDTH
IMG_CHANNELS = convert_to_records.IMG_CHANNELS
IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS

TRAIN_DATA_DIR='../data/train/'
TEST_DATA_DIR='../data/test/'
TFR_SAVE_DIR='../data/'
NUM_TRAIN =2000
NUM_VALI = 1000

IMG_PIXELS2 = 350*350*3;


def read_and_decode(filename_queue):
    #params: filename， tensorflow中的队列，pop出tfrecords里记录的文件名
    #使用tensorflow队列高速读取文件,是符号类型，在sess.run()时才启动
    #根据文件名生成一个队列
    #filename_queue = tf.train.string_input_producer([filename])

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
    #img = tf.reshape(img, [224, 224, 3])
    img.set_shape([IMG_PIXELS])
    img = tf.reshape(img, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

def read_tfr_queue(tfr_fn, batch_size, num_epochs=DEFAULT_EPOCH_NUM_LIMIT, shuffle=DEFAULT_EPOCH_SHUFFLE):
    '''
    产生batch_size的数据，以tensorflow的队列形式, 需要在session中启动，每run一次，出一个batch
    :param tfr_fn: tfrecords文件路径及文件名
    :param batch_size:
    :param num_epochs: 限制队列产出所有数据轮数，若不设置，若无数轮。
    :param shuffle: 若为true, 每换一次epochs，都打乱一下数据
    :return:batch数据
    '''
    with tf.name_scope('read_tfr_queue'):
        filename_queue = tf.train.string_input_producer(
            [tfr_fn],num_epochs=num_epochs,shuffle=shuffle)


    img, label = read_and_decode(filename_queue)
    # min_after_dequeue,出队列后，队列最小有多长。与train数据、batch_size有关。capacity要比min_aft... 要大至少3个batch_size
    # bacth_size，一次出8张，为随机的一个集合
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size, capacity=2000,
                                                    min_after_dequeue=1000,num_threads=2)

    return img_batch, label_batch

def test_read_tfrecords():
    #启动tf.Session后，tfrecords将不断pop出文件名
    tfr_fn='../data/train_shuffle.tfrecords'#'data_small/train.tfrecords'
    #img, label = read_and_decode(tfr_fn)
    # 使用shuffle_batch可以随机打乱输入
    #img_batch, label_batch = tf.train.shuffle_batch([img, label],
    #                                                batch_size=8, capacity=2000,
    #                                                min_after_dequeue=1000)
    img_batch_queue, label_batch_queue = read_tfr_queue(tfr_fn,8)
    # min_after_dequeue,出队列后，队列最小有多长。与train数据、batch_size有关。如果有tfr train有1000张dog，1000张cat，只各取500张出来训
    # bacth_size，一次出8张，为随机的一个集合
    init = tf.initialize_all_variables()
    #因为使用了num_epochs
    init_local= tf.initialize_local_variables()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_local)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #threads = tf.train.start_queue_runners(sess=sess)
        try:
            #for i in range(3):
            step=0
            step_wanted=1
            while(not coord.should_stop()) and (step < step_wanted):
                step+=1
                val, l = sess.run([img_batch_queue, label_batch_queue])
                # 我们也可以根据需要对val， l进行处理
                # l = to_categorical(l, 12)
                print(val.shape, l)
                if((1 in l) and (0 in l) ):

                    for j in range(3):
                        Image.fromarray(((val[j] + 0.5) * 255).astype(np.uint8)).show()
        except tf.errors.OutOfRangeError:
            print('Done pop out all data in the tfrecords' )
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