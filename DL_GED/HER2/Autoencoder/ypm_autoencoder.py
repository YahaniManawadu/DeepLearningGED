
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np


from preprocessData_cnn import create_test_and_test_data
train_x, train_y, test_x, test_y = create_test_and_test_data()


# Parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 1000
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = len(train_x[0]) # data input
n_classes = 2 # total classes


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    i = 0

    # Loop over all batches
    while i < len(train_x):
        start = i
        end = i + batch_size

        batch_x = np.array(train_x[start: end])
        batch_y = np.array(train_y[start: end])
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
    # Display logs per epoch step
        if i % display_step == 0:
            print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(c))

        i += batch_size

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(y_pred, feed_dict={X: test_x, y: test_y})

    # Test model
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(X, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:",accuracy)
    print("Accuracy:", accuracy.eval({X: encode_decode}))

