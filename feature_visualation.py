# import layer as layer
import tensorflow
# import numpy as np
# import random
# from keras.datasets import cifar10
# from keras.utils import np_utils
# from keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import img_to_array, load_img
# from keras.models import Sequential
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers.core import Dense, Flatten, Activation, Dropout
# import matplotlib.pyplot as plt

from keras_applications.densenet import layers
# from tensorflow_core.contrib.timeseries.python.timeseries import model
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


def visual_image(path):
    print(' model')

    batch_size = 50
    features = 32 * 32
    categories = 2
    filter1 = 32
    filter2 = 64

    train_path = r'dataset2\train\[0-1]'
    test_path = r'dataset2\test\[0-1]'
    validation_path = r'dataset2\validation\[0-1]'



    data_x = dataX(features, train_path)
    print("datax: ", data_x)
    data_y = dataY(categories, train_path)
    print("datay: ", data_y)
    data_x_test = dataX(features, test_path)
    data_y_test = dataY(categories, test_path)
    data_x_validation = dataX(features, validation_path)
    data_y_validation = dataY(categories, validation_path)

    x_image = tensorflow.reshape(data_x, [-1, 32, 32, 1])
    x_image_test = tensorflow.reshape(data_x_test, [-1, 32, 32, 1])
    x_image_validation = tensorflow.reshape(data_x_validation, [-1, 32, 32, 1])

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
                  loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(x=x_image, y=data_y, epochs=10,
                        validation_data=(x_image_validation, data_y_validation))

    # Show results in graph view
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.show()

    test_loss, test_acc = model.evaluate(x_image_test, data_y_test, verbose=2)

    # plt.show()
# def visual_image(path):

    # model= tensorflow.keras.models.Sequential()
    # model.add(tensorflow.keras.layers.InputLayer(input_shape=(32, 32 , 3)))
    # model.add(tensorflow.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='SAME'))
    # model.add(tensorflow.keras.layers.MaxPooling2D((2, 2), padding='SAME', strides=(2, 2)))
    # model.add(tensorflow.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='SAME'))
    # model.add(tensorflow.keras.layers.MaxPooling2D((2, 2), padding='SAME', strides=(2, 2)))
    # model.add(tensorflow.keras.layers.Flatten())
    # model.add(tensorflow.keras.layers.Dense(64, activation='relu'))
    # model.add(tensorflow.keras.layers.Dropout(0.5))
    # model.add(tensorflow.keras.layers.Dense(3))
    # model.compile(optimizer='adam',
    #               loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    layer_outputs = [layer.output for layer in model.layers[1:]]
    visualize_model = tensorflow.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    img = load_img(path, target_size=(32, 32))
    # x = img_to_array(img)
    # print(x.shape)
    img = img.convert('L')  # convert image to 8-bit grayscale
    data = list(img.getdata())
    x = np.array(data)
    x = x.reshape(1, 32, 32,1)
    print(x.shape)
    x = x/255

    feature_maps = visualize_model.predict(x)
    print(len(feature_maps))
    layer_names=[layer.name for layer in model.layers]
    print(layer_names)
    # layer_names = ['cov1', 'maxpool1', 'cov2', 'maxpool2', 'fc1', 'fc2']
    for layer_names, feature_maps in zip(layer_names,feature_maps):
        print(feature_maps.shape)
        if len(feature_maps.shape) == 4:
            channels = feature_maps.shape[-1]
            size = feature_maps.shape[1]
            display_grid = np.zeros((size, size * channels))
            for i in range(channels):
                x = feature_maps[0, :, :, i]
                x -= x.mean()
                # print(x.std())
                x /= (x.std()+0.001)
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size : (i + 1) * size] = x
            scale = 20. / channels
            plt.figure(figsize=(scale * channels, scale))
            plt.title(layer_names)
            plt.grid(False)

            plt.imshow(display_grid, aspect= 'auto', cmap='viridis')
            plt.show()


# def create_cnn_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding = 'same', input_shape=X_train.shape[1:]))
#     model.add(Activation('relu'))
#
#     # Adding more layers to improve the model
#     model.add(Conv2D(32, (3, 3), padding = 'same'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size= (2,2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(64, (3, 3), padding = 'same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size= (2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))
#     return model


if __name__ == '__main__':
    visual_image("dataset2/train/1/00000_00027.jpg")
