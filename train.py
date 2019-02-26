from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCNN

import os
import sys

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcnn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 700, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 80, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 80, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 80, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 90, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden6', 90, 'Number of units in hidden layer 6.')
flags.DEFINE_integer('hidden7', 28, 'Number of units in hidden layer 7.')
flags.DEFINE_integer('hidden8', 28, 'Number of units in hidden layer 8.')
flags.DEFINE_integer('hidden9', 30, 'Number of units in hidden layer 9.')
flags.DEFINE_integer('hidden10', 24, 'Number of units in hidden layer 10.')
flags.DEFINE_integer('hidden11', 24, 'Number of units in hidden layer 11.')
flags.DEFINE_integer('hidden12', 24, 'Number of units in hidden layer 12.')
flags.DEFINE_integer('node_output_size', 20, 'Number of hidden features each node has prior to readout')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# command line arguments
flags.DEFINE_integer('random_seed',12,'random seed for repeatability')
flags.DEFINE_string('data_input_path',sys.argv[1],'data path')
flags.DEFINE_string('dir_model','models/','directory for storing saved models')
flags.DEFINE_string('output_name','unnamed_model','name of the saved model')
flags.DEFINE_string('input_name',None,'name of the saved model')

# flags for saving and loading data
flags.DEFINE_bool('should_load_previous_data', False, 'should we load data from a previous execution?')
flags.DEFINE_string('data_output_path', None, 'data_output_path')

# Set random seed
seed = FLAGS.random_seed
np.random.seed(seed)
tf.set_random_seed(seed)

# make sure directories are properly set
if not os.path.exists(FLAGS.dir_model):
    os.makedirs(FLAGS.dir_model)
if FLAGS.data_input_path[-1] is not "/":
  FLAGS.data_input_path += "/"
if FLAGS.data_output_path is not None:
  if not os.path.exists(FLAGS.data_output_path):
      os.makedirs(FLAGS.data_output_path)
  if not FLAGS.data_output_path[-1]=="/":
      FLAGS.data_output_path += "/"

# get model executable. Add your own model!
if FLAGS.model == 'gcnn':
    model_constructor = GCNN
# if FLAGS.model == 'your model':
#     model_constructor = your_model
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

def train(data, placeholders):
    start_time = time.time()
    feed_dict = construct_feed_dict(data, placeholders)
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.mae], feed_dict=feed_dict)
    end_time = time.time()
    train_outputs = {
        'loss': outs[1],
        'accuracy': outs[2],
        'targets': outs[3],
        'mean_absolute_error': outs[4],
        'duration': (end_time - start_time)
    }
    return train_outputs

def evaluate(data, placeholders):
    start_time = time.time()
    feed_dict = construct_feed_dict(data, placeholders)
    outs = sess.run([model.loss, model.accuracy, model.mae], feed_dict=feed_dict)
    end_time = time.time()
    evaluation_outputs = {
        'loss': outs[0],
        'accuracy': outs[1],
        'mean_absolute_error': outs[2],
        'duration': (end_time - start_time)
    }
    return evaluation_outputs

def initialize_tensorboard_outputs():
    train_loss = tf.Variable(0.)
    train_acc = tf.Variable(0.)
    
    a = tf.summary.scalar("Train Loss", train_loss)
    b = tf.summary.scalar("Train Acc", train_acc)
    
    summary_ops = tf.summary.merge([a, b])

    visualization_data = {
      'train_loss': train_loss,
      'train_accuracy': train_acc,
    }
    return visualization_data, summary_ops


#############
# Load data #
#############
# data contains: 
# adj
# features
# y_train
# y_val
# y_test
# train_mask
# val_mask
# test_mask
# molecule_partitions
# num_molecules
###############

start_time = time.time()
data = load_data(FLAGS.data_input_path, FLAGS.data_output_path, FLAGS.should_load_previous_data)
end_time = time.time()
####
print("Finished loading data in {} seconds".format(end_time-start_time))

# Some feature preprocessing and other important quantities
preprocessed_features = preprocess_features(data['features'])
adjacency_matrices = preprocess_adj(data['adj'])
num_adj_matrices = len(data['adj'])

# Define placeholders
placeholders = {
  'adjacency_matrices': [tf.sparse_placeholder(tf.float32) for _ in range(num_adj_matrices)],
  'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(preprocessed_features[2], dtype=tf.int64)),
  'labels': tf.placeholder(tf.float32, shape=(None,data['y_train'].shape[1]), name='the_labels'),
  'labels_mask': tf.placeholder(tf.int32, name='the_mask_of_labels'),
  'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
  'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
  'molecule_partitions': tf.placeholder(tf.int32),
  'num_molecules': tf.placeholder(tf.int32,shape=())
}

# Define inputs for training/testing/validation sets
train_inputs = {
  'features': preprocessed_features,
  'adjacency_matrices': adjacency_matrices,
  'labels': data['y_train'],
  'labels_mask': data['train_mask'],
  'molecule_partitions': data['molecule_partitions'],
  'num_molecules': data['num_molecules'],
  'dropout': FLAGS.dropout,
}

validation_inputs = {
  'features': preprocessed_features,
  'adjacency_matrices': adjacency_matrices,
  'labels': data['y_val'],
  'labels_mask': data['val_mask'],
  'molecule_partitions': data['molecule_partitions'],
  'num_molecules': data['num_molecules'],
}

testing_inputs = {
  'features': preprocessed_features,
  'adjacency_matrices': adjacency_matrices,
  'labels': data['y_test'],
  'labels_mask': data['test_mask'],
  'molecule_partitions': data['molecule_partitions'],
  'num_molecules': data['num_molecules'],
}

# Create model
model = model_constructor(placeholders,input_dim=preprocessed_features[2][1], logging=True) # input_dim is number of features per node

# Setup tensorboard!
print("Initializing session......")

saver = tf.train.Saver()
sess = tf.Session()

# Tensorboard stuff
visualization_data, visualization_operation = initialize_tensorboard_outputs()
visualization_writer = tf.summary.FileWriter('tensorboard/' + FLAGS.output_name + '/', sess.graph)

# Init variables
print("Initializing variables......")
sess.run(tf.global_variables_initializer())

# If the user requested to retrain from an existing model, load it now 
if FLAGS.input_name is not None:
    saver.restore(sess,FLAGS.dir_model+FLAGS.input_name+'/'+FLAGS.input_name)

# Normalizes labels in model
print("Normalizing LABELS......")
if FLAGS.input_name is None:
    [m,s]=sess.run([model.get_mean, model.get_std], feed_dict={placeholders['labels']: data['y_train'], placeholders['labels_mask']: data['train_mask']})

# Training
validation_losses = []
total_training_time=0.
for epoch in range(FLAGS.epochs):

    # Training step
    train_outputs = train(train_inputs, placeholders)

    # Validation
    validation_outputs = evaluate(validation_inputs, placeholders)

    # save validation loss so we can see if we need to stop early
    validation_losses.append(validation_outputs['loss'])

    # Print results
    print("Epoch: ", '%04d' % (epoch + 1),
          "train_loss: ", "{:.5f}".format(train_outputs['loss']),
          "val_loss: ", "{:.5f}".format(validation_outputs['loss']),
          "train_acc: ", str(train_outputs['accuracy']),        
          "val_acc: ", str(validation_outputs['accuracy']),
          "train_mae: ", str(train_outputs['mean_absolute_error']),
          "time: ", "{:.5f}".format(train_outputs['duration']))

    # Keep track of time consumed
    total_training_time += train_outputs['duration']

    # Log data to tensorboard for visualization
    summary = sess.run(visualization_operation, feed_dict={
        visualization_data['train_loss']: train_outputs['loss'], #training loss
        visualization_data['train_accuracy']: train_outputs['accuracy'] #training acc
    })
    visualization_writer.add_summary(summary, epoch)

    # check if we need to early stop, check if the average of the last Flags.early_stopping losses are less than the latest
    if epoch > FLAGS.early_stopping and validation_losses[-1] > np.mean(validation_losses[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# save model in case we want to load it in the future
saver.save(sess, FLAGS.dir_model+FLAGS.output_name+'/'+FLAGS.output_name)
visualization_writer.flush()

# Run evaluation on the testing set to see how we did!
testing_outputs = evaluate(testing_inputs, placeholders)

# Print results
print("Test set results: ",
      "loss: ", "{:.5f}".format(testing_outputs['loss']),
      "accuracy: ", str(testing_outputs['accuracy']),
      "mae: ", str(testing_outputs['mean_absolute_error']))

print("Train set results:",
      "loss: ","{:.5f}".format(train_outputs['loss']),
      "accuracy: ", str(train_outputs['accuracy']),
      "mae: ",str(train_outputs['mean_absolute_error']))

print("Vald set results:", 
      "loss: ","{:.5f}".format(validation_outputs['loss']),
      "accuracy: ", str(validation_outputs['accuracy']),
      "mae: ",str(validation_outputs['mean_absolute_error']))

print("time: ", "{:.5f}".format(total_training_time), "s")


def log_results():
  '''Write results to file'''
  return
