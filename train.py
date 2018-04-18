from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils2 import *
from gcn.models import JCNN

import os

#usage
#python train.py --output_name my_new_model_name --input_name my_restored_model_name

#load_distros('../cleaned/')
#load_distros('../shuffled/1000/')




# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
red = 20
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'jcnn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 700, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 100-red, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 80-red, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 80-red, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('hidden4', 80-red, 'Number of units in hidden layer 4.')
flags.DEFINE_integer('hidden5', 90-red, 'Number of units in hidden layer 5.')
flags.DEFINE_integer('hidden6', 90-red, 'Number of units in hidden layer 6.')
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
flags.DEFINE_string('data_path','error','data path')
flags.DEFINE_string('dir_model','models/','directory for storing saved models')
flags.DEFINE_string('output_name','unnamed','name of the saved model')
flags.DEFINE_string('input_name',None,'name of the saved model')

flags.DEFINE_integer('load_previous',0,'load_previous')
flags.DEFINE_string('pklpath','error!!!','pklpath')


def build_summaries():
    train_loss = tf.Variable(0.)
    train_acc = tf.Variable(0.)

    a = tf.summary.scalar("Train Loss", train_loss)
    b = tf.summary.scalar("Train Acc", train_acc)

    summary_vars = [train_loss,train_acc]
    summary_ops = tf.summary.merge([a,b])
    summary_writer = tf.summary.FileWriter('tensorboard/'+FLAGS.output_name+'/',sess.graph)

    return summary_ops, summary_vars, summary_writer


if not os.path.exists(FLAGS.dir_model):
    os.makedirs(FLAGS.dir_model)

# Set random seed
seed = FLAGS.random_seed
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
load_previous = FLAGS.load_previous
pickle=1
#pklpath = '10kdipole/'
pklpath = FLAGS.pklpath
if not os.path.exists('./loaded_data/' + pklpath):
    os.makedirs('./loaded_data/' + pklpath)

[adj,features,y_train,y_val,y_test,train_mask,val_mask,test_mask,molecule_partitions,num_molecules]=load_data3(FLAGS.data_path,pklpath,pickle,load_previous)
#[adj_new,features_new,y_new,molecule_partitions_new,num_molecules_new]=load_data_new(data_path_new)

#support_new = [preprocess_adj(adj_new)]
#features_new = preprocess_features(features_new)

print("Finished loading data!")


# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'jcnn':
    support = preprocess_adj(adj)
    num_supports = len(adj)
    model_func = JCNN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None,y_train.shape[1]), name='the_labels'),
    'labels_mask': tf.placeholder(tf.int32, name='the_mask_of_labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout_meow'),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'molecule_partitions': tf.placeholder(tf.int32),
    'num_molecules': tf.placeholder(tf.int32,shape=())
}


# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
#input_dim is like...if you have k features for each node, then input_dim=k


# Define model evaluation function
def evaluate(features, support, labels, molecule_partitions, num_molecules, placeholders, mask=None):
    if mask is None:
        mask = np.array(np.ones(labels.shape[0]), dtype=np.bool)
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, molecule_partitions, num_molecules, placeholders)
    outs_val = sess.run([model.loss, model.accuracy,model.mae], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


# Initialize session
print("Initializing session......")
saver = tf.train.Saver()
sess = tf.Session()

# Tensorboard stuff
summary_ops, summary_vars, summary_writer = build_summaries()

# Init variables
print("Initializing variables......")
sess.run(tf.global_variables_initializer())

if FLAGS.input_name is not None:
    saver.restore(sess,FLAGS.dir_model+FLAGS.input_name+'/'+FLAGS.input_name)


# feed_dict = construct_feed_dict(features, support, y_train, train_mask, molecule_partitions, num_molecules, placeholders)
    
# savior = sess.run([model.getmae],feed_dict=feed_dict)
# print(savior)
# exit()

#normalize targets in model
print("Normalizing targets......")

if FLAGS.input_name is None:
    [m,s]=sess.run([model.get_mean,model.get_std], feed_dict={placeholders['labels']: y_train, placeholders['labels_mask']: train_mask})

cost_val = []


for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, molecule_partitions, num_molecules, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,model.outputs,model.mae], feed_dict=feed_dict)

    summary_str = sess.run(summary_ops, feed_dict={
        summary_vars[0]: outs[1], #training loss
        summary_vars[1]: outs[2] #training acc
    })
    summary_writer.add_summary(summary_str, epoch)

    # Validation
    cost, acc, mae, duration = evaluate(features, support, y_val, molecule_partitions, num_molecules, placeholders, mask=val_mask)
    cost_val.append(cost)

    # if (epoch == 40):
    #     FLAGS.learning_rate = 1.0
    #     print_learn_rate(FLAGS.learning_rate)
    # if (epoch == 100):
    #     FLAGS.learning_rate = 0.5        
    #     print_learn_rate(FLAGS.learning_rate)
    # if (epoch == 150):
    #     FLAGS.learning_rate = 0.1
    #     print_learn_rate(FLAGS.learning_rate)
    # if (epoch == 200):
    #     FLAGS.learning_rate = 0.05        
    #     print_learn_rate(FLAGS.learning_rate)
    # if (epoch == 250):
    #     FLAGS.learning_rate = 0.01
    #     print_learn_rate(FLAGS.learning_rate)
    # if (epoch == 300):
    #     FLAGS.learning_rate = 0.005
    #     print_learn_rate(FLAGS.learning_rate)
    # if (epoch == 500):
    #     FLAGS.learning_rate = 0.001
    #     print_learn_rate(FLAGS.learning_rate)

    # Log a summary ever 10 steps
    #if epoch % 10 == 0:
    #    summary_writer.add_summary(some_kind_of_summary, epoch)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "val_loss=", "{:.5f}".format(cost), "train_acc= ", str(outs[2]),        
          " val_acc= ", str(acc),"train_mae= ", str(outs[4]), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break


print("Optimization Finished!")


saver.save(sess,FLAGS.dir_model+FLAGS.output_name+'/'+FLAGS.output_name)
summary_writer.flush()

# Testing
test_cost, test_acc, test_mae, test_duration = evaluate(features, support, y_test, molecule_partitions, num_molecules, placeholders,mask=test_mask)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy= ", str(test_acc), "mae= ", str(test_mae)) #, "time=", "{:.5f}".format(test_duration))

print("Tran set results:", "cost=","{:.5f}".format(outs[1]),
      "accuracy= ", str(outs[2]), "mae= ",str(outs[4]))

print("Vald set results:", "cost=","{:.5f}".format(cost),
      "accuracy= ", str(acc), "mae= ",str(mae))


# Costs_file = open("wat.txt","a+")
# Costs_file.write(pklpath + "\n")
# Costs_file.write("Epochs: " + str(epoch + 1) + "\ntrain_loss= " + str(outs[1]) + 
#     "      val_loss= " + str(cost) + "\ntrain_acc= " + str(outs[2]) + "\nval_acc= " + 
#     str(acc) + "\ntrain_mae= " + str(outs[5]))
# Costs_file.write("\n")
# Costs_file.write("Test cost= " + str(test_cost) + "\nTest_acc= " +str(test_acc))
# Costs_file.write("\nTest mae= " + str(test_mae))
# Costs_file.write("\n\n")
# Costs_file.close()