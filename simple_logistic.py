# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
import glob

# tf.compat.v1.disable_resource_variables()
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

    learning_rate = 0.00001
    training_epochs = 10000
    batch_size = 100
    display_step = 1
    features = 32 * 32
    categories = 4
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, categories])
    W = tf.Variable(tf.zeros([features, categories]))
    b = tf.Variable(tf.zeros([categories]))
    z = tf.matmul(x, W) + b
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y_, z))

    update = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
    data_x = dataX(features, r'dataset2\train\[0-3]')
    print("datax: ", data_x)
    data_y = dataY(categories, r'dataset2\train\[0-3]')
    print("datay: ", data_y)
    data_x_test = dataX(features, r'dataset2\test\[0-3]')
    data_y_test = dataY(categories, r'dataset2\test\[0-3]')
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(0, 500):
        total_batch = int(len(data_x) / batch_size)
        for j in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size,data_x,data_y)
            sess.run(update, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            print("Iteration:", i, ",      Loss: ", loss.eval(session=sess, feed_dict = {x:data_x, y_:data_y}))

    # Test model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(session=sess, feed_dict = {x: data_x_test, y_: data_y_test}))

    for i in range(len(data_x_test)):
        print('Prediction for: "', data_x_test[i], '": ', sess.run(y, feed_dict={x: [data_x_test[i]]}), ',  Max value: ', max(sess.run(y, feed_dict={x: [data_x_test[i]]})[0]), ',  Sum: ', sum(sess.run(y, feed_dict={x: [data_x_test[i]]})[0]), ', real class: ', data_y_test[i])


if __name__ == '__main__':
    model()
