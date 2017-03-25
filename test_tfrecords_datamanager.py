#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
先运行convert_to_recondes.py 产生tfr文件

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import maybe_download
import GetInputData as gid
import convert_to_records
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

cwd = os.getcwd()

IMG_HEIGHT = convert_to_records.IMG_HEIGHT
IMG_WIDTH = convert_to_records.IMG_WIDTH
IMG_CHANNELS = convert_to_records.IMG_CHANNELS
IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS




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
    #img = tf.reshape(img, [224, 224, 3])
    img.set_shape([IMG_PIXELS])
    img = tf.reshape(img, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

def test_read_tfrecords():
    #启动tf.Session后，tfrecords将不断pop出文件名
    tfr_fn='data_small/train.tfrecords'
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
        #threads = tf.train.start_queue_runners(sess=sess)
        try:
            #for i in range(3):
            step=0
            step_wanted=100
            while(not coord.should_stop()) and (step < step_wanted):
                step+=1
                val, l = sess.run([img_batch, label_batch])
                # 我们也可以根据需要对val， l进行处理
                # l = to_categorical(l, 12)

                if((1 in l) and (0 in l) ):
                    print(val.shape, l)
                    for j in range(3):
                        Image.fromarray(((val[j] + 0.5) * 255).astype(np.uint8)).show()
        except tf.errors.OutOfRangeError:
            print('Done pop out all data in the tfrecords' )
        finally:
            coord.request_stop()
            print("DEBUG: try to finally")
        coord.join(threads)


if __name__ == "__main__":
    test_read_tfrecords()
