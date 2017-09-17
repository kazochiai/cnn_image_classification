import tensorflow as tf
import unit_tests.problem_unittests as tests
import helper.helper as helper

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    #return tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]], name='x')
    return tf.placeholder(tf.float32, shape=[None, *image_shape], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """

    return tf.placeholder(tf.float32, shape=[None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """

    return tf.placeholder(tf.float32, shape=None, name='keep_prob')


"""
Tests
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

"""
Convolution layer
"""
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # kernel and bias
    filter_size_height = conv_ksize[0]
    filter_size_width = conv_ksize[1]
    x_shape = x_tensor.shape.as_list()
    x_depth = x_shape[3]

    conv_init_values = tf.truncated_normal([filter_size_height, filter_size_width, x_depth, conv_num_outputs],
                                           stddev=0.1)
    kernel = tf.Variable(conv_init_values)
    bias = tf.constant(0.1)#tf.Variable(tf.zeros(conv_num_outputs))

    # Apply Convolution
    conv_layer = tf.nn.conv2d(x_tensor, kernel, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME',
                              data_format='NHWC')
    # Add bias
    conv_layer_bias = conv_layer + bias  # tf.nn.bias_add(conv_layer, bias)
    # Apply activation function
    conv_layer_activated = tf.nn.relu(conv_layer_bias)

    # Set the ksize (filter size) for each dimension (batch_size, height, width, depth)
    ksize = [1, pool_ksize[0], pool_ksize[1], 1]
    max_pool = tf.nn.max_pool(conv_layer_activated, ksize, strides=[1, pool_strides[0], pool_strides[1], 1],
                              padding='SAME')

    return max_pool

    # conv = tf.layers.conv2d(x_tensor, conv_num_outputs, conv_ksize, strides=conv_strides, padding='SAME')
    # maxp = tf.contrib.layers.max_pool2d(conv, pool_ksize, stride=pool_strides, padding='SAME')
    # return maxp


"""
Tests
"""
tests.test_con_pool(conv2d_maxpool)


"""
Flatten layer
"""
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    # shape is (?, 10, 30, 6), want to make it (?, 1800)
    shape = x_tensor.shape.as_list()
    reshaped = tf.reshape(x_tensor, [-1, shape[1] * shape[2] * shape[3]])
    return reshaped
    #tf.contrib.layers.flatten(x_tensor) #reshaped


"""
Test
"""
tests.test_flatten(flatten)

"""
Fully connected layer
"""
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    x_shape = x_tensor.shape.as_list()
    x_shape_row = x_shape[1]

    weight = tf.Variable(tf.truncated_normal([x_shape_row, num_outputs], stddev=0.1))
    bias = tf.constant(0.1)#tf.Variable(tf.zeros(num_outputs))

    out = tf.add(tf.matmul(x_tensor, weight), bias)
    activated = tf.nn.relu(out)
    return activated


# tf.contrib.layers.fully_connected(x_tensor, num_outputs)

"""
Test
"""
tests.test_fully_conn(fully_conn)


"""
Output layer
"""


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    x_shape = x_tensor.shape.as_list()
    x_shape_row = x_shape[1]
    weight = tf.Variable(tf.truncated_normal([x_shape_row, num_outputs], stddev=0.1))
    bias = tf.constant(0.1)#tf.Variable(tf.zeros(num_outputs))

    out = tf.add(tf.matmul(x_tensor, weight), bias)
    return out

"""
Test
"""
tests.test_output(output)

"""
Create Convolutional Model
"""
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_num_outputs = 64
    conv_ksize = (4, 4)
    conv_strides = (1, 1)
    pool_ksize = (2, 2)
    pool_strides = (2, 2)
    conv1 = conv2d_maxpool(x, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)

    conv_num_outputs2 = 128
    conv_ksize2 = (2, 2)
    conv_strides2 = (2, 2)
    pool_ksize2 = (4, 4)
    pool_strides2 = (1, 1)
    conv2 = conv2d_maxpool(conv1, conv_num_outputs2, conv_ksize2, conv_strides2, pool_ksize2, pool_strides2)

    conv_num_outputs3 = 256
    conv_ksize3 = (2, 2)
    conv_strides3 = (1, 1)
    pool_ksize3 = (2, 2)
    pool_strides3 = (1, 1)
    conv3 = conv2d_maxpool(conv2, conv_num_outputs3, conv_ksize3, conv_strides3, pool_ksize3, pool_strides3)

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    flattened_x = flatten(conv3)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    f_conn = fully_conn(flattened_x, 1000)
    f_conn_d = tf.nn.dropout(f_conn, keep_prob)
    f_conn2 = fully_conn(f_conn_d, 600)
    f_conn2_d = tf.nn.dropout(f_conn2, keep_prob)
    f_conn3 = fully_conn(f_conn2_d, 200)

    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    out = output(f_conn3, 10)
    # out = tf.nn.softmax(out)
    # TODO: return output
    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)

"""
Train the Neural Network
"""
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """

    # Run optimizer and get loss
    session.run([optimizer, cost], feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})


"""
Test
"""
tests.test_train_nn(train_neural_network)


"""
Show stats
"""
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    global valid_features
    global valid_labels
    # TODO: Implement Function

    cost_dict = {x: feature_batch, y: label_batch, keep_prob: 1}
    cost_value = session.run(cost, feed_dict=cost_dict)

    validation_feed_dict = {x: valid_features, y: valid_labels, keep_prob: 1}
    validation_accuracy = session.run(accuracy, feed_dict=validation_feed_dict)

    print('loss:{0}, validation accuracy:{1}'.format(cost_value, validation_accuracy))


"""
Hyperparameters
"""
epochs = 100
batch_size = 1000
keep_probability = 0.70

import pickle
"""
Train on a Single CIFAR-10 Batch¶
"""

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# print('Checking the Training on a Single Batch...')
# with tf.Session() as sess:
#     # Initializing the variables
#     sess.run(tf.global_variables_initializer())
#
#     # Training cycle
#     for epoch in range(epochs):
#         batch_i = 1
#         for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
#             train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
#         print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
#         print_stats(sess, batch_features, batch_labels, cost, accuracy)


save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)