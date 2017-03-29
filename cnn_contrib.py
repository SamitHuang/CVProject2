'''
#CNN for dogs_vs_cats
#by  HUANG, Yongixang
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import GetInputData

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 16
NUM_TRAIN_DATA = GetInputData.NUM_TRAIN
NUM_EPOCH = 40

#MODEL_DIR = 

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, GetInputData.IMAGE_SIZE_WIDTH,GetInputData.IMAGE_SIZE_HEIGTH, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1_1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  conv1_2 = tf.layers.conv2d(
      inputs=conv1_1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2_1 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  conv2_2 = tf.layers.conv2d(
      inputs=conv2_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

  #3
  conv3_1 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  conv3_2 = tf.layers.conv2d(
      inputs=conv3_1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2, 2], strides=2)


  # 4
  conv4_1 = tf.layers.conv2d(
      inputs=pool3,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  conv4_2 = tf.layers.conv2d(
      inputs=conv4_1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  shape = int(np.prod(pool4.get_shape()[1:]))  # except for batch size (the first one), multiple the dimensions
  pool4_flat = tf.reshape(pool4, [-1, shape])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense1 = tf.layers.dense(inputs=pool4_flat, units=256, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

  dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu)
  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.5, training=mode == learn.ModeKeys.TRAIN)


      # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout2, units=2)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.00005,
        optimizer="Adam")#"SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
  # Load training and eval data
  #mnist = learn.datasets.load_dataset("mnist")
  
  #train_data = mnist.train.images  # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  train_data,train_labels,eval_data,eval_labels = GetInputData.GetTrainAndValidateData()
  train_data = np.asarray(train_data,dtype=np.float32)
  #train_data = train_data /255.0 - 0.5;
  eval_data = np.asarray(eval_data,dtype=np.float32)
  #eval_data = eval_data/255.0 - 0.5;
  train_labels = np.asarray(train_labels,dtype=np.int32)
  eval_labels = np.asarray(eval_labels, dtype=np.int32)
  print(train_data.shape)
  
  validation_metrics = {
      "accuracy_eval": MetricSpec(
          metric_fn=tf.contrib.metrics.streaming_accuracy,
          prediction_key="classes"),
      "precision_val": MetricSpec(
          metric_fn=tf.contrib.metrics.streaming_precision,
          prediction_key="classes")
  }
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      eval_data,
      eval_labels,
      every_n_steps=50,
      metrics=validation_metrics,
      early_stopping_metric="loss",
      early_stopping_metric_minimize=True,  #True
      early_stopping_rounds=1000)
  #eval_data = mnist.test.images  # Returns np.array

  # Create the Estimator
  classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/dog_cat_model_pix128"
      ,config=tf.contrib.learn.RunConfig(save_checkpoints_secs=4))
  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50,)
  
  # Train the model
  classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=BATCH_SIZE,
      steps=NUM_TRAIN_DATA * NUM_EPOCH / BATCH_SIZE,
      monitors=[validation_monitor])
  
  print("number of total train step=%d"%(NUM_TRAIN_DATA * NUM_EPOCH / BATCH_SIZE))
  
  # Configure the accuracy metric for evaluation

  metrics = {
      "accuracy_final":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  cnt1=0
  cnt0=0
  for i in train_labels:
      if (i==1) :
          cnt1+=1
      if(i==0):
          cnt0+=1
  print(cnt1,cnt0)
  eval_results = classifier.evaluate(
          x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)

def test_predict():
    test_data,img_ids=GetInputData.GetTestData()
    with open('submission_file.csv','w') as f:
        f.write('id,label\n')
    #print(img_id)
    classifier = learn.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/dog_cat_model_pix128"
        )
    #there are 12500 images in the test dataset 
    PRED_BATCH_SIZE=100
    div_step = int(math.ceil(float(len(img_ids))/PRED_BATCH_SIZE))
    for i in  range(div_step):
        offset=i * PRED_BATCH_SIZE
        if(offset + PRED_BATCH_SIZE <= len(img_ids)):
            img_ids_step = img_ids[offset : offset + PRED_BATCH_SIZE]
            preds = list(classifier.predict(test_data[offset : offset + PRED_BATCH_SIZE]))
        else:
            img_ids_step = img_ids[offset:]
            preds = list(classifier.predict(test_data[offset : ]))
        write_csv(img_ids_step,preds)
        
def write_csv(img_ids,preds):
    with open('submission_file.csv','a') as f:
        for (i,res_one) in enumerate(preds):
            img_id=img_ids[i]
            #prob=round(res_one['probabilities'][1],8)
            prob=res_one['probabilities'][1]
            classes = res_one['classes']
            print(img_id,classes)
            #f.write('{},{}\n'.format(img_id,prob))
            f.write('%d,%.8f\n'%(img_id,prob))


if __name__ == "__main__":
  #tf.app.run()
  test_predict()
