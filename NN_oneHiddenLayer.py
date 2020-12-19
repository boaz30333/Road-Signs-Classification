# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np
from numpy import asarray
from PIL import Image
from numpy import clip
import glob
import random as rn
import os
import shutil


def show_image():
    # load the image
    image = Image.open('sydney_bridge.jpg')
    print(' show_image()')
    # summarize some details about the image
    print(image.format)
    print(image.mode)
    print(image.size)
    # show the image
    # image.show()

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

def dataX(features):
    data_x = np.array([])
    count = 0
    for filepath in glob.iglob(r'dataset2\train\[0-2]'):
        path = filepath.split("\\")
        globpath = filepath + '\*.jpg'
        for filepath in glob.iglob(r'' + globpath):
            count = count + 1
            img = Image.open(filepath).convert('L')  # convert image to 8-bit grayscale
            data = list(img.getdata())
            x = np.array(data)
            data_x = np.append(data_x, x)
            #print("x: ", x)
    print("count", count)
    data_x = data_x.reshape(count, features)
    #print("data_x: ", data_x)
    return data_x.astype(int)
    #return data_x
def dataY(categories):
    data_y = np.array([])
    count = 0
    for filepath in glob.iglob(r'dataset2\train\[0-2]'):
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
    #print("data_y: ", data_y)
    return data_y.astype(int)
    #return data_y

def dataXTest(features):
    data_x = np.array([])
    count = 0
    for filepath in glob.iglob(r'dataset2\test\[0-2]'):
        path = filepath.split("\\")
        globpath = filepath + '\*.jpg'
        for filepath in glob.iglob(r'' + globpath):
            count = count + 1
            img = Image.open(filepath).convert('L')  # convert image to 8-bit grayscale
            data = list(img.getdata())
            x = np.array(data)
            data_x = np.append(data_x, x)
            #print("x: ", x)
    #print("count", count)
    data_x = data_x.reshape(count, features)
    #print("data_x: ", data_x)
    return data_x.astype(int)
def dataYTest(categories):
    data_y = np.array([])
    count = 0
    for filepath in glob.iglob(r'dataset2\test\[0-2]'):
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
    #print("data_y: ", data_y)
    return data_y.astype(int)


def model():
    #image = Image.open(path)
    print(' model')
    #pixels = asarray(image)

    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1
    features = 32 * 32
    categories = 3
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, categories])
    W = tf.Variable(tf.random_normal([features, categories]))
    W = tf.Variable(tf.zeros([features, categories]))
    b = tf.Variable(tf.random_normal([categories]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    z = tf.matmul(x, W) + b
    data_x = dataX(features)
    print("datax: ", data_x)
    data_y = dataY(categories)
    print("datay: ", data_y)
    data_x_test = dataXTest(features)
    data_y_test = dataYTest(categories)
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(3609 / batch_size)
            # Loop over all batches
            sess.run(W)
            sess.run(b)
            for i in range(total_batch):
                batch_xs, batch_ys = next_batch(batch_size,data_x,data_y)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:","W: ",W , "b: ",b ,  '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: data_x_test, y: data_y_test}))

    y = tf.nn.softmax(z)
    loss = -tf.reduce_mean(y_ * tf.log(y))
    #loss = tf.reduce_mean(loss1)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( y_, z))

    update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)



    #sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    #for i in range(0, 1000):
       # sess.run(update, feed_dict={x: data_x, y_: data_y})
        #if i % 10 == 0:
            #print(("W: ", W))
            #print("B: ", b )
         #   print("loss: ", loss.eval(session=sess, feed_dict={x: data_x, y_: data_y}))
            #print("loss", loss)
            #print("update: ", update)
    #for i in range(len(data_x_test)):
     #   print('Prediction for: "' + data_x_test[i] + ': "', sess.run(y, feed_dict={x: [data_x_test[i]]}))

    #basewidth = 10
    #wpercent = (basewidth / float(img.size[0]))
    #hsize = int((float(img.size[1]) * float(wpercent)))
    #img = img.resize((basewidth, hsize), Image.ANTIALIAS)


      # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)


    # At this point the image's pixels are all in memory and can be accessed
    # individually using data[row][col].


    #print(a)
    #for row in data:
        #print(' '.join('{:3}'.format(value) for value in row))
    #img.show()
    #np.save(file_name, a)
    #np.savetxt(x+"/"+file_name+".csv", a, delimiter=',', fmt='%d')


def pick_random_20_precent(path, lib_name):
    count=0
    globpath= path+'\*.jpg'
    for filepath in glob.iglob(r''+globpath):
        count+=1
    print(count)
    print(lib_name)
    precent= count*0.2
    source_dir = 'road signs\dataset2\\train\\'+lib_name
    target_dir1 = 'road signs\dataset2\\test\\'+lib_name
    target_dir2='road signs\dataset2\\validation\\'+lib_name
    #file_names = os.listdir(source_dir)

   # for file_name in file_names:
    #    shutil.move(os.path.join(source_dir, file_name), target_dir)
    while precent>=1:
        random_file = rn.choice(os.listdir(source_dir))
        shutil.move(os.path.join(source_dir, random_file), target_dir1)
        random_file = rn.choice(os.listdir(source_dir))
        shutil.move(os.path.join(source_dir, random_file), target_dir2)
        precent-=1


def Normalize_Pixel_Values():
    # load image
    image = Image.open('sydney_bridge.jpg')
    print(' Normalize_Pixel_Values()')
    pixels = asarray(image)
    print(pixels)
    # fh = open("test6.txt", "w")
    # pixels.tofile(fh)
    arr_reshaped = pixels.reshape(pixels.shape[0], -1)
    #np.savetxt('data.csv', arr_reshaped, delimiter=',', fmt='%d')
    #np.savetxt('try.txt', arr_reshaped, delimiter=',', fmt='%d')
    np.save('try2', pixels)
    # print(pixels)
    # confirm pixel range is
    print('Data Type: %s' % pixels.dtype)
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # with open('try.txt', 'w') as file:
    #   file.write(pixels.tolist())

    # (pixels.tofile('try.txt', " "))
    # confirm the normalization
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

def load_npz():
    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    print('load_npz()')
    path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']
        print(train_examples)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
        print(test_dataset)

        BATCH_SIZE = 64
        SHUFFLE_BUFFER_SIZE = 100

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['sparse_categorical_accuracy'])
        model.fit(train_dataset, epochs=10)

        model.evaluate(test_dataset)

def Global_Centering():
    # load image
    image = Image.open('sydney_bridge.jpg')
    print('Global_Centering()')
    pixels = asarray(image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate global mean
    mean = pixels.mean()
    print('Mean: %.3f' % mean)
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    # global centering of pixels
    pixels = pixels - mean
    # confirm it had the desired effect
    mean = pixels.mean()
    arr_reshaped=pixels.reshape(pixels.shape[0], -1)
    np.savetxt('data2.csv', arr_reshaped, delimiter=',', fmt='%d')
    print('Mean: %.3f' % mean)
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))


def Local_Centering():
    # load image
    image = Image.open('sydney_bridge.jpg')
    print('Local_Centering()')
    pixels = asarray(image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate per-channel means and standard deviations
    print('////////')
    # print(pixels.tolist())
    means = pixels.mean(axis=(0, 1), dtype='float64')
    print('Means: %s' % means)
    print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0, 1)), pixels.max(axis=(0, 1))))
    # per-channel centering of pixels
    pixels -= means
    print('////////')
    print(pixels)
    # confirm it had the desired effect
    means = pixels.mean(axis=(0, 1), dtype='float64')
    print('Means: %s' % means)
    print('Mins: %s, Maxs: %s' % (pixels.min(axis=(0, 1)), pixels.max(axis=(0, 1))))


def Global_Standardization():
    # load image
    image = Image.open('sydney_bridge.jpg')
    print('Global_Standardization()')
    pixels = asarray(image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate global mean and standard deviation
    mean, std = pixels.mean(), pixels.std()
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    # confirm it had the desired effect
    mean, std = pixels.mean(), pixels.std()
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))


def Positive_Global_Standardization():
    # load image
    image = Image.open('sydney_bridge.jpg')
    print('Positive_Global_Standardization()')
    pixels = asarray(image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate global mean and standard deviation
    mean, std = pixels.mean(), pixels.std()
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    # clip pixel values to [-1,1]
    pixels = clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0
    # confirm it had the desired effect
    mean, std = pixels.mean(), pixels.std()
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))


def Local_Standardization():
    # load image
    image = Image.open('sydney_bridge.jpg')
    print('Local_Standardization()')
    pixels = asarray(image)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate per-channel means and standard deviations
    means = pixels.mean(axis=(0, 1), dtype='float64')
    stds = pixels.std(axis=(0, 1), dtype='float64')
    print('Means: %s, Stds: %s' % (means, stds))
    # per-channel standardization of pixels
    pixels = (pixels - means) / stds
    # confirm it had the desired effect
    means = pixels.mean(axis=(0, 1), dtype='float64')
    stds = pixels.std(axis=(0, 1), dtype='float64')
    print('Means: %s, Stds: %s' % (means, stds))


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    model()
    #show_image()
    #Normalize_Pixel_Values()
    #Global_Centering()
    #Local_Centering()
    #Global_Standardization()
    #Positive_Global_Standardization()
    #Local_Standardization()
    #load_npz()
    #for filepath in glob.iglob(r'road signs\dataset2\train\*'):
       # x = filepath.split("\\")
        #if x[3]=='I-1' or x[3]=='I-2' or x[3]=='I-3':
        #if x[3] != '1':
        #parent_dir = 'road signs/dataset2/validation/'
        #path = os.path.join(parent_dir, x[3])
         #   pick_random_20_precent(filepath, x[3])
        #os.mkdir(path)
            #continue
        #print(x[3])
        #os.mkdir(x[3])
        #pick_random_20_precent(filepath,x[3])
        #pick_random_20_precent('road signs\dataset\\train\I-3', 'I-3')
    #for filepath in glob.iglob(r'ts\*\*.png'):
     #   x=filepath.split("\\")
      #  print(x[1])
       # print(len(x[1]))
        #print(filepath[3+len(x[1])+1:-4])
        #print(filepath)
        #print(len(filepath))
        #print(filepath[7:-4])
        #from_png_to_npy(filepath,filepath[3+len(x[1])+1:-4], x[1])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
