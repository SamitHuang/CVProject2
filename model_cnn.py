#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#import read_data
import numpy as np
import tensorflow as tf
import math

BATCH_SIZE =16
NUM_TRAIN_EXAMPLES = 32
NUM_TRAIN_STEP = int(math.ceil(NUM_TRAIN_EXAMPLES/BATCH_SIZE))

#model parameters
NUM_CLASSES = 2
DROP_PROB = 0.5
REG_STRENGTH = 0.001
INITIAL_LEARNING_RATE = 0.001
LR_DECAY_FACTOR = 0.5
EPOCHS_PER_LR_DECAY = 5
MOVING_AVERAGE_DECAY = 0.9999




def _activation_summary(x):
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='reg_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inference(imgs):
    '''
    Forward的网络结构，不包括输出层
    :param imagesInputPlaceholder:
    :return:
    '''
    # conv 1
    with tf.variable_scope('conv1') as scope:
        weights = _variable_with_weight_decay('weights', shape=[5, 5, 3, 32], stddev=1 / np.sqrt(5 * 5 * 3), wd=0.00)
        #weights = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32,
        #                                    stddev=1 / np.sqrt(5 * 5 * 3)), name='weights_conv1')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(imgs, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # conv 2
    with tf.variable_scope('conv2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[5, 5, 32, 64], stddev=1 / np.sqrt(5 * 5 * 32), wd=0.00)
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv1, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # pool 1
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv 3
    with tf.variable_scope('conv3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1 / np.sqrt(3 * 3 * 64), wd=0.00)
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)

    # conv 4
    with tf.variable_scope('conv4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1 / np.sqrt(3 * 3 * 64), wd=0.00)
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv3, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)

    # pool 2
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv 5
    with tf.variable_scope('conv5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1 / np.sqrt(3 * 3 * 64), wd=0.00)
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv5)

    # conv 6
    with tf.variable_scope('conv6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1 / np.sqrt(3 * 3 * 64), wd=0.00)
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv5, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv6)

    # pool 3
    with tf.variable_scope('pool3') as scope:
        pool3 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fully connected 1
    with tf.variable_scope('fc1') as scope:
        batch_size = imgs.get_shape()[0].value #imagesInputPlaceholder.get_shape()[0].value
        pool3_flat = tf.reshape(pool3, [batch_size, -1])
        dim = pool3_flat.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=1 / np.sqrt(dim), wd=REG_STRENGTH)
        biases = tf.get_variable('biases', shape=[384], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(pool3_flat, weights) + biases, name=scope.name)
        _activation_summary(fc1)

    # fully connected 2
    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=1 / np.sqrt(384), wd=REG_STRENGTH)
        biases = tf.get_variable('biases', shape=[192], initializer=tf.constant_initializer(0.0))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
        _activation_summary(fc2)

        # dropout
        fc2_drop = tf.nn.dropout(fc2, DROP_PROB)

    # Softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', shape=[192, 2], stddev=1 / np.sqrt(192), wd=0.000)
        biases = tf.get_variable('biases', shape=[2], initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fc2_drop, weights), biases, name=scope.name)
        _activation_summary(logits)

    return logits


def loss(logits, labels):
    '''
    定义模型的损失函数
    :param logits: model_inference的输出
    :param labels:
    :return:
    '''
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    data_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', data_loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss

def _loss_summaries(total_loss):
    losses = tf.get_collection('losses')
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)

def training(total_loss):
    '''
    定义训练网络的方法，即优化器，minibatch/RMProp/Adam,此处用Adam
    :param total_loss:
    :return:
    '''
    global_step = tf.Variable(0, name='global_step', trainable=False)
    decay_steps = int(EPOCHS_PER_LR_DECAY * NUM_TRAIN_EXAMPLES / BATCH_SIZE)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LR_DECAY_FACTOR, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    _loss_summaries(total_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    opt_op = optimizer.minimize(total_loss, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    mov_average_object = tf.train.ExponentialMovingAverage(0.9999, global_step)
    moving_average_op = mov_average_object.apply(tf.trainable_variables())

    with tf.control_dependencies([opt_op]):
        train_op = tf.group(moving_average_op)

    return train_op

def evaluation(logits, true_labels):
    correct_pred = tf.nn.in_top_k(logits, true_labels, 1)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))*100
