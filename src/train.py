import time
import tensorflow as tf
import numpy as np
import networkx as nx

import data_utils
from model import *

print("TF Version: ", tf.__version__)

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataname', 'twitter', 'Dataset string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('regularization_scale', 0.001, 'Scale for L2 regularization.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('hidden_dim', 100, 'Hidden Dimension')
flags.DEFINE_integer('max_steps', 30, 'Number of steps in RNN.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

# Load data
G, node_index, train_examples, train_mask, test_examples, test_mask = data_utils.load_data('datasets/' + FLAGS.dataname, FLAGS.max_steps + 1)

indices = np.random.permutation(train_examples.shape[0])
to_index = int(len(indices) * 0.9)
training_idx, validation_idx = indices[:to_index], indices[to_index:]
train_examples, validation_examples = train_examples[training_idx, :], train_examples[validation_idx, :]
train_mask, validation_mask = train_mask[training_idx, :], train_mask[validation_idx, :]

number_of_nodes = len(G.nodes())

print ("Number of nodes: {}".format(number_of_nodes))
print ("Number of edges: ", len(G.edges()))
print ("Number of training examples: {}".format(len(train_examples)))
print ("Number of validation examples: {}".format(len(validation_examples)))
print ("Number of testing examples: {}".format(len(test_examples)))
print ("Shape of train_examples: {}".format(train_examples.shape))
print ("Shape of train_mask: {}".format(train_mask.shape))
print ("Shape of validation_examples: {}".format(validation_examples.shape))
print ("Shape of validation_mask: {}".format(validation_mask.shape))
print ("Shape of test_examples: {}".format(test_examples.shape))
print ("Shape of test_mask: {}".format(test_mask.shape))
print ("Average train cascade size: {}".format(np.mean(np.sum(train_mask, axis=1))))
print ("Average validation cascade size: {}".format(np.mean(np.sum(validation_mask, axis=1))))
print ("Average test cascade size: {}".format(np.mean(np.sum(test_mask, axis=1))))

print ("***** Hyper Parameters *****")
print ("Learning rate: {}".format(FLAGS.learning_rate))
print ("Batch size: {}".format(FLAGS.batch_size))
print ("Max steps: {}".format(FLAGS.max_steps))
print ("Regularization scale: {}".format(FLAGS.regularization_scale))
print ("hidden_dim: {}".format(FLAGS.hidden_dim))

train_batches = data_utils.Loader(train_examples, train_mask, FLAGS.batch_size)
print ("Number of train batches: {}".format(len(train_batches)))

# Define placeholders
placeholders = {
    'contents': tf.placeholder(tf.float32, shape=(None, FLAGS.hidden_dim)),
    'sequences': tf.placeholder(tf.int32, shape=(None, FLAGS.max_steps + 1)),
    'seq_mask': tf.placeholder(tf.int32, shape=(None, FLAGS.max_steps)),
    'hit_at': tf.placeholder(tf.int32)
}

# Create model
model = CascadeRNN(number_of_nodes, FLAGS.hidden_dim, FLAGS.max_steps, nx.to_numpy_matrix(G).astype(np.float32), placeholders, name='cascadernn', logging=True)

# Initialize session
sess = tf.Session()

def evaluate(eval_feed_dict):
    eval_feed_dict[placeholders['hit_at']] = 10
    hitat_10, loss = sess.run([model.metric, model.loss], feed_dict=eval_feed_dict)
    eval_feed_dict[placeholders['hit_at']] = 50
    hitat_50 = sess.run(model.metric, feed_dict=eval_feed_dict)
    eval_feed_dict[placeholders['hit_at']] = 100
    hitat_100 = sess.run(model.metric, feed_dict=eval_feed_dict)
    return hitat_10, hitat_50, hitat_100, loss

# Init variables
sess.run(tf.global_variables_initializer())

val_loss = []
# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()

    for batch_num in range(len(train_batches)):
        batch_examples, batch_mask = train_batches()
        feed_dict = {
            placeholders['contents']: np.zeros((batch_examples.shape[0], FLAGS.hidden_dim)),
            placeholders['sequences']: batch_examples, 
            placeholders['seq_mask']: batch_mask,
            placeholders['hit_at']: 10
        }
        _, train_loss, train_hitat_10 = sess.run([model.opt_op, model.loss, model.metric], feed_dict=feed_dict)
    
    val_feed_dict = {
        placeholders['contents']: np.zeros((validation_examples.shape[0], FLAGS.hidden_dim)),
        placeholders['sequences']: validation_examples,
        placeholders['seq_mask']: validation_mask
    }

    hitat_10, hitat_50, hitat_100, loss = evaluate(val_feed_dict)

    print ("Epoch {:04d} (time={:.5f}): train_loss={:.5f} train_hit@10={:.5f} validation_loss={:.5f} validation_hit@10={:.5f} validation_hit@50={:.5f} validation_hit@100={:.5f}".format(epoch + 1, time.time() - t, train_loss, train_hitat_10, loss, hitat_10, hitat_50, hitat_100))
    val_loss.append(loss)
    if epoch > FLAGS.early_stopping and val_loss[-1] > np.mean(val_loss[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_t = time.time()
test_feed_dict = {
    placeholders['contents']: np.zeros((test_examples.shape[0], FLAGS.hidden_dim)),
    placeholders['sequences']: test_examples,
    placeholders['seq_mask']: test_mask
}
test_hit_10, test_hit_50, test_hit_100, test_cost = evaluate(test_feed_dict)
print("*** Test set results: *** \ntime={:.5f} test_loss={:.5f} test_hit@10={:.5f} test_hit@50={:.5f} test_hit@100={:.5f}".format((time.time() - test_t), test_cost, test_hit_10, test_hit_50, test_hit_100))
