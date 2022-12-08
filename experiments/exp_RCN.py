import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior
tf.disable_eager_execution()

log_dir1='log'
raw_training_data_dir = 'data/raw_data/train_data'
process_data_dir = 'data/process_data'
experiment_name = 'RCN'
main_model_root = 'model/network_model'
checkpoint_model_dir = os.path.join(main_model_root, experiment_name)

n_slice = 21
image_size = 112
train_batch_size=10
validation_batch_size=10
segt_class=4
learning_rate=0.001
lr_decay_rate=0.99
train_epoch=201
continue_training=False

