# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob


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
    # return data_x
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
            t = [int(path[2])]
            y = np.append(y, t)
            data_y = np.append(data_y, y)
    # data_y = data_y.reshape(count, categories)
    # return data_y
    return data_y.astype(int)


def model():
    print(' model')

    batch_size = 50
    features = 32 * 32
    categories = 5
    filter1 = 32
    filter2 = 64

    train_path = r'dataset\train\[0-4]'
    test_path = r'dataset\test\[0-4]'
    validation_path = r'dataset\validation\[0-4]'



    data_x = dataX(features, train_path)
    print("datax: ", data_x)
    data_y = dataY(categories, train_path)
    print("datay: ", data_y)
    data_x_test = dataX(features, test_path)
    data_y_test = dataY(categories, test_path)
    data_x_validation = dataX(features, validation_path)
    data_y_validation = dataY(categories, validation_path)

    x_image = tf.reshape(data_x, [-1, 32, 32, 1])
    x_image_test = tf.reshape(data_x_test, [-1, 32, 32, 1])
    x_image_validation = tf.reshape(data_x_validation, [-1, 32, 32, 1])

    model = models.Sequential()
    model.add(layers.Conv2D(filter1, (5, 5), activation='relu', padding='SAME', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2), padding='SAME', strides=(2, 2)))
    model.add(layers.Conv2D(filter2, (5, 5), activation='relu', padding='SAME', input_shape=(filter1, 16, 16, 1)))
    model.add(layers.MaxPooling2D((2, 2), padding='SAME', strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(filter2, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(categories))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(x=x_image, y=data_y, epochs=10,
                        validation_data=(x_image_validation, data_y_validation))

    # Show results in graph view
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    # plt.show()

    test_loss, test_acc = model.evaluate(x_image_test, data_y_test, verbose=2)

    plt.show()

if __name__ == '__main__':
    model()
