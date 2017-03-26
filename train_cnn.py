#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import math
import os.path
#import read_data
import model_cnn
import tensorflow as tf
import eval_cnn
import DataManager


TRAIN_DATA_DIR = 'tmp/train_data/'
CHECKPOINT_FILE = 'model.ckpt'
CHECKPOINT_FILE_PATH = os.path.join(TRAIN_DATA_DIR, CHECKPOINT_FILE)
BATCH_SIZE = model_cnn.BATCH_SIZE

NUM_EPOCHS = 3  #暂时没用

NUM_TRAIN_EXAMPLES = model_cnn.NUM_TRAIN_EXAMPLES
NUM_TRAIN_STEP = model_cnn.NUM_TRAIN_STEP


def run_training():
    with tf.Graph().as_default():
        #train_images, train_labels = read_data.inputs(data_set='train', batch_size=BATCH_SIZE, num_epochs=1)
        train_images, train_labels = DataManager.read_tfr_queue(DataManager.TFR_SAVE_DIR + 'train_unshuffle.tfrecords',BATCH_SIZE)
        train_logits = model_cnn.inference(train_images)
        train_accuracy = model_cnn.evaluation(train_logits, train_labels)
        tf.summary.scalar('train_accuracy', train_accuracy)

        loss = model_cnn.loss(train_logits, train_labels)

        train_op = model_cnn.training(loss)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        summary_op = tf.summary.merge_all()

        init_op = tf.initialize_all_variables()

        sess = tf.Session()

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(TRAIN_DATA_DIR, sess.graph)

        try:
            step = 0
            num_iter_per_epoch = int(math.ceil(NUM_TRAIN_EXAMPLES / BATCH_SIZE))

            while (not coord.should_stop()) and (step < NUM_TRAIN_STEP):
                start_time = time.time()

                _, loss_value, train_acc_val = sess.run([train_op, loss, train_accuracy])

                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 2 == 0:
                    print('Step %d : loss = %.5f , training accuracy = %.1f (%.3f sec)'
                          % (step, loss_value, train_acc_val, duration))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                if step % num_iter_per_epoch == 0 and step > 0: # Do not save for step 0
                    num_epochs = int(step / num_iter_per_epoch)
                    saver.save(sess, CHECKPOINT_FILE_PATH, global_step=step)
                    print('epochs done on training dataset = %d' % num_epochs)
                    eval_cnn.evaluate('validation', checkpoint_dir=TRAIN_DATA_DIR)

                step += 1

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps' % (NUM_EPOCHS, step))
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()