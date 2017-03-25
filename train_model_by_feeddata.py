#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import GetInputData as gid
import model_cnn
import tensorflow as tf
import time
import math
import DataManager


#model parameter
MODEL_NAME ="simple1" # 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')
LR = 0.001 #learning rate
BATCH_SIZE = 16
NUM_EPOCHS = 2

def accuracy(predictions, labels):
   return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def TrainModel():
    # 建好graph, 在graph里组织好各op、数据流等
    graph=tf.Graph()
    with graph.as_default():
        #input
        #trainX,trainY,valX,valY = gid.GetTrainAndValidateData()
        #train_images, train_labels = read_data.inputs(data_set='train', batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS)
        #train_dataset=tf.placeholder(
        #    tf.float32, shape=(BATCH_SIZE, gid.IMAGE_SIZE_HEIGTH, gid.IMAGE_SIZE_WIDTH, 3))
        #train_labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
        #validation_dataset=tf.placeholder(
         #   tf.float32, shape=(BATCH_SIZE, gid.IMAGE_SIZE_HEIGTH, gid.IMAGE_SIZE_WIDTH, 3))
        #validation_labels=tf.placeholder(tf.int32, shape=(BATCH_SIZE, 2))
        train_images, train_labels = DataManager.read_tfr_queue(DataManager.TFR_SAVE_DIR + 'train_shuffle.tfrecords',
                                                                BATCH_SIZE)

        #inference op
        train_logits = model_cnn.inference(train_images)
        #accuracy op
        train_accuracy = model_cnn.evaluation(train_logits, train_labels)
        #tf.scalar_summary('train_accuracy', train_accuracy)

        loss1 = model_cnn.loss(train_logits, train_labels)

        train_op = model_cnn.training(loss1)

        #validation_logits = model_cnn.inference(validation_dataset);
        #valdation_accuracy = model_cnn.evaluation(validation_logits, validation_labels)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        summary_op = tf.summary.merge_all()



        # init_op = tf.initialize_all_variables()

    #建session，开始跑graph
    sess=tf.Session(graph=graph)
    with sess as session:
        tf.initialize_all_variables().run()
        tf.initialize_local_variables().run()
        summary_writer = tf.summary.FileWriter("./log", sess.graph)
        #print "global variables are initialized"
        #灌数据, 训练和验证
        #trainX,trainY,testX,testY=gid.GetTrainAndValidateData()
        #trainY=trainY[:,0]
        #print(trainX.shape)

        #feed_dict = {train_images: trainX, train_labels: trainY}
        start_time = time.time()
        _, loss_value,train_acc_val = sess.run([train_op, loss1,train_accuracy], feed_dict=feed_dict)
        duration = time.time() - start_time
        #train_acc_val = accuracy(loss_value, trainY);
        print('Step %d : loss = %.5f , training accuracy = %.1f (%.3f sec)\n'
              % (1, loss_value, train_acc_val, duration))
        '''
        numStepPerEpoch = int(math.ceil(gid.NUM_TRAIN/NUM_EPOCHS));
        for ei in range(NUM_EPOCHS):
            for si in range(numStepPerEpoch):
                offset = si * BATCH_SIZE;
                batchX = trainX[offset:offset+BATCH_SIZE,:,:,:];
                batchY = trainY[offset:offset+BATCH_SIZE]
                feed_dict = {train_dataset: batchX, train_labels: batchY}
                #跑graph中的op,train op， loss op
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss1], feed_dict=feed_dict)
                duration = time.time() - start_time
                train_acc_val = accuracy(loss_value,trainY);
                if si % 2 == 0:

                    print('Step %d : loss = %.5f , training accuracy = %.1f (%.3f sec)\n'
                          % (si, loss_value, train_acc_val, duration))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, si)
                    #val_acc = sess.run([valdation_accuracy],feed_dict={validation_dataset:testX,validation_labels:testY})
                    #print('validation acc = %.2f' % (val_acc))

        '''

if __name__ == "__main__":
    TrainModel()
    #img=Image.open('../data/train/cat.0.jpg').convert('RGB').resize((2, 2))
    #img.show()
    #print img.shape
    #dat = np.array(img)
    #dat = np.asarray(img) #.transpose(2,0,1) 若是用Theano 或者 Caffe，需要transpose以符合输入格式;
    #print dat.shape
    #print dat
