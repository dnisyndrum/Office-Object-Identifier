from keras.models import Sequential
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import MaxPooling2D, AvgPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.regularizers import l2
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from ann_visualizer.visualize import ann_viz
import tensorflow as tf


height, width = 32, 32
continue_training = True
LOF, MOF, HOF, VHOF = 1, 3, 5, 7     # low order features, medium order features, high order features, very high
channels = 3
pooling_size = 2
output_classes = 4
batch_size = 3
steps_per_epoch = 1669
validation_steps = 400
epochs = 150


def create_model():
    my_model = Sequential()

    my_model.add(Conv2D(32, (MOF, MOF), input_shape=(height, width, channels), padding='same'))
    my_model.add(BatchNormalization())
    my_model.add(Activation('relu'))

    my_model.add(AvgPool2D(pool_size=(pooling_size, pooling_size)))

    my_model.add(Conv2D(64, (MOF, MOF), padding='same'))
    my_model.add(BatchNormalization())
    my_model.add(Activation('relu'))

    my_model.add(AvgPool2D(pool_size=(pooling_size, pooling_size)))

    my_model.add(Conv2D(128, (MOF, MOF), padding='same'))
    my_model.add(BatchNormalization())
    my_model.add(Activation('relu'))

    my_model.add(AvgPool2D(pool_size=(pooling_size, pooling_size)))

    my_model.add(Conv2D(256, (MOF, MOF), padding='same'))
    my_model.add(BatchNormalization())
    my_model.add(Activation('relu'))

    my_model.add(AvgPool2D(pool_size=(pooling_size, pooling_size)))

    my_model.add((Flatten()))

    my_model.add(Dense(512, activation='relu'))
    my_model.add(Dropout(0.5))

    my_model.add(Dense(output_classes, activation='softmax'))

    my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return my_model


def train_validate_model(my_model):
    classes = ['ir_camera', 'ir_camera_battery', 'overhead_light', 'power_supply']

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2
    )

    training_set = train_datagen.flow_from_directory(
        'ir_dataset/train',
        target_size=(height, width),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    validation_set = train_datagen.flow_from_directory(
        'ir_dataset/validation',
        target_size=(height, width),
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=True
    )

    history = my_model.fit_generator(
        training_set,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=validation_set
    )

    print('Model score: ')
    score = my_model.evaluate_generator(validation_set, steps=100)

    print("Loss: ", score[0], "Accuracy: ", score[1])

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return my_model


def save(my_model):
    my_model.save('ir_ident_model.h5')


def load():
    return load_model('ir_ident_model.h5')


def predict(my_model):

    images_list = ['ir_dataset/test/img1.jpg', 'ir_dataset/test/img2.jpg', 'ir_dataset/test/img3.jpg',
                   'ir_dataset/test/img4.jpg', 'ir_dataset/test/img5.jpg', 'ir_dataset/test/img6.jpg',
                   'ir_dataset/test/img7.jpg', 'ir_dataset/test/img8.jpg', 'ir_dataset/test/img9.jpg',
                   'ir_dataset/test/img10.jpg', 'ir_dataset/test/img11.jpg', 'ir_dataset/test/img12.jpg',
                   'ir_dataset/test/img13.jpg', 'ir_dataset/test/img14.jpg', 'ir_dataset/test/img15.jpg',
                   'ir_dataset/test/img16.jpg', 'ir_dataset/test/img17.jpg', 'ir_dataset/test/img18.jpg',
                   'ir_dataset/test/img19.jpg', 'ir_dataset/test/img20.jpg']

    for img in images_list:
        cur_img = image.load_img(img, target_size=(height, width))
        temp = image.img_to_array(cur_img)
        temp = np.expand_dims(temp, axis=0)
        vstack = np.vstack([temp])
        predict_this = my_model.predict_classes(vstack, batch_size=1)
        print(predict_this)

    print('expected: 0, 3, 1, 1, 2, 2, 2, 0, 0, 3, 1, 2, 0, 1, 3, 0, 2, 0, 3, 2')


if os.path.exists('ir_ident_model.h5'):
    print('Existing model found')
    model = load()
    print('Model loaded')
    if continue_training:
        model = train_validate_model(model)
        save(model)
else:
    print('No existing model present, creating/training new model')
    model = create_model()
    mode = train_validate_model(model)
    save(model)
    print('Model saved')

predict(model)
model.summary()





































