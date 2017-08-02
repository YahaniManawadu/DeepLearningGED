
# By : Y.P. Manawadu
# Research : Deep Learning Analysis of gene expression data for Breast Cancer Classification
# Code : to execute the deep learning Algorithm, Convolutional Neural Network

import tensorflow as tf
import numpy as np

# Import data
from preprocessData_cnn import create_test_and_test_data
train_x, train_y, test_x, test_y = create_test_and_test_data()


# Parameters
learning_rate = 0.001
training_iters = 100
batch_size = 10
display_step = 10

# Network Parameters
n_classes = 2 # total classes

#To reduce overfitting, we will apply dropout before the readout layer
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, len(train_x[0])])
y = tf.placeholder(tf.float32, [None, n_classes])

#placeholder for the probability that a neuron's output is kept during dropout.
#this  allows us to turn dropout on during training, and turn it off during testing.
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):#convolutions uses a stride of one and are zero padded so that the output is the same size as the input
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):#plain old max pooling over 2x2 blocks
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')



# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input  #1st dimension reshape x to a 4d tensor  # 2nd and 3rd dimensions corresponding to iinput width and height,
    x = tf.reshape(x, shape=[-1, 4, 4, 1]) #[-1, 28, 28, 1]  # 1=number of channels.

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    #You can get the shape of the tensor x using x.get_shape().as_list().
    # For getting the first dimension (batch size) you can use x.get_shape().as_list()[0]

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    # tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them
    #so dropout just works without any additional scaling
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out']) # multiply by a weight matrix, add a bias, and apply a ReLU.
    return out

# Store layers weight & bias
weights = {
    #[5, 5, 1, 32] # 5x5 conv, 1 input, 32 outputs #convolution will compute 32 features for each 5x5 patch
    'wc1': tf.Variable(tf.random_normal([4,4, 1, 2])),#1=number of input channels #2=number of output channels #first two dimensions are the patch size
     #[5, 5, 32, 64] 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([4,4, 2, 4])),

    #[7*7*64, 1024] # fully connected, 7*7*64 inputs, 1024 outputs
    #now input size has been reduced to 7x7
    #we add a fully-connected layer with 1024 neurons to allow processing on the entire image.
    'wd1': tf.Variable(tf.random_normal([1*2*2, 8])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([8, n_classes]))
}

biases = {# bias vector with a component for each output channel.
    'bc1': tf.Variable(tf.random_normal([2])),
    'bc2': tf.Variable(tf.random_normal([4])),
    'bd1': tf.Variable(tf.random_normal([8])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

def tf_confusion_metrics(session, feed_dict):

    predictions = tf.argmax(pred, 1)
    actuals = tf.argmax(y, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    tp, tn, fp, fn = session.run([tp_op, tn_op, fp_op, fn_op],feed_dict)

    tpr = float(tp) / (float(tp) + float(fn))
    fpr = float(fp) / (float(tp) + float(fn))

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    precision = float(tp) / (float(tp) + float(fp))

    f1_score = (2 * (precision * recall)) / (precision + recall)

    print ('Precision = ', precision)
    print ('Recall = ', recall)
    print ('F1 Score = ', f1_score)
    print ('Accuracy = ', accuracy)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Loop over all batches
    i = 0
    while i < len(train_x):
        start = i
        end = i + batch_size

        batch_x = np.array(train_x[start: end])
        batch_y = np.array(train_y[start: end])
    
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        i += batch_size

    print("Optimization Finished!")

    # Calculate accuracy for test data
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.}))
    tf_confusion_metrics(sess, feed_dict={x: test_x, y: test_y, keep_prob: 1.})