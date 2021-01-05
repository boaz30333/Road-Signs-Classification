# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
import glob

tf.compat.v1.disable_resource_variables()
tf.disable_v2_behavior()




def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def dataX(features, set):
    data_x = np.array([])
    count = 0
    for filepath in glob.iglob(set):
        globpath = filepath + '\*.jpg'
        for filepath in glob.iglob(r'' + globpath):
            count = count + 1
            img = Image.open(filepath).convert('L')  # convert image to 8-bit grayscale
            data = list(img.getdata())
            x = np.array(data)
            data_x = np.append(data_x, x)
    data_x = data_x.reshape(count, features)
    return data_x.astype(int)

def dataY(categories, set):
    data_y = np.array([])
    count = 0
    for filepath in glob.iglob(set):
        path = filepath.split("\\")
        globpath = filepath + '\*.jpg'
        for filepath in glob.iglob(r'' + globpath):
            count = count + 1
            y = np.array([])
            for i in range(categories):
                if i != int(path[2]):
                    y = np.append(y, [0])
                else:
                    y = np.append(y, [1])
            data_y = np.append(data_y, y)
    data_y = data_y.reshape(count, categories)
    return data_y.astype(int)


def model():
    print(' model')

    batch_size = 50
    features = 32 * 32
    categories = 3
    filter1 = 32
    filter2 = 64
    hidden_layer_nodes_1 = 100
    hidden_layer_nodes_2 = 50
    path = r'dataset2\train\[0-2]'

    data_x = dataX(features, path)
    print("datax: ", data_x)
    data_y = dataY(categories, path)
    print("datay: ", data_y)
    data_x_test = dataX(features, path)
    data_y_test = dataY(categories, path)
    data_x_validation = dataX(features, path)
    data_y_validation = dataY(categories, path)

    x = tf.placeholder(tf.float32, shape=[None, features])
    y_ = tf.placeholder(tf.float32, shape=[None, categories])

    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, filter1], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[filter1]))
    x_image = tf.reshape(x, [-1, 32, 32, 1])  # if we had RGB, we would have 3 channels

    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, filter1, filter2], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[filter2]))

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * filter2])
    W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * filter2, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal([1024, categories], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[categories]))

    z = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    y_conv = tf.nn.softmax(z)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_, z))
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # uses moving averages momentum
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(4000):
        batch_xs, batch_ys = next_batch(batch_size, data_x, data_y)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x: data_x_test, y_: data_y_test, keep_prob: 1.0}))




if __name__ == '__main__':
    model()
