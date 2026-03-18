import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, Lambda, BatchNormalization, Activation, Input
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import pandas as pd

BATCH_SIZE = 32
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
NUM_CLASSES = 2
DATA_DIR = 'dataset'
VALID_DIR = os.path.join(DATA_DIR, 'valid')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def create_my_model(input_shape, num_classes):
    model = models.Sequential(name='MyCNN')
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))

    return model


def create_alexnet(input_shape, num_classes):
    model = models.Sequential(name='AlexNet')
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))

    return model


def create_vgg16(input_shape, num_classes, trainable_base=False):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = trainable_base

    model = models.Sequential(name='VGG16')
    model.add(layers.Input(shape=input_shape))
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))

    return model


def create_vgg19(input_shape, num_classes, trainable_base=False):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = trainable_base

    model = models.Sequential(name='VGG19')
    model.add(layers.Input(shape=input_shape))
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))

    return model


def residual_block(x, number_of_filters, match_filter_size=False, initializer=HeNormal()):
    x_skip = x
    if match_filter_size:
        x = layers.Conv2D(number_of_filters, (3, 3), strides=(2, 2),
                          kernel_initializer=initializer, padding='same')(x_skip)
    else:
        x = layers.Conv2D(number_of_filters, (3, 3), strides=(1, 1),
                          kernel_initializer=initializer, padding='same')(x_skip)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = layers.Conv2D(number_of_filters, (3, 3), kernel_initializer=initializer, padding='same')(x)
    x = BatchNormalization()(x)
    if match_filter_size:
        x_skip = Lambda(lambda z: tf.pad(z[:, ::2, ::2, :],
                                         [[0, 0], [0, 0], [0, 0],
                                          [number_of_filters // 4, number_of_filters // 4]]))(x_skip)
    x = Add()([x, x_skip])
    x = Activation('relu')(x)

    return x


def create_resnet(input_shape, num_classes, n=3):
    initializer = HeNormal()
    init_filters = 16
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(init_filters, (3, 3), strides=(1, 1), kernel_initializer=initializer, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    filter_size = init_filters
    for layer_group in range(3):
        for block in range(n):
            if layer_group > 0 and block == 0:
                filter_size *= 2
                x = residual_block(x, filter_size, match_filter_size=True, initializer=initializer)
            else:
                x = residual_block(x, filter_size, initializer=initializer)
    x = GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, kernel_initializer=initializer)(x)
    model = Model(inputs=inputs, outputs=x, name=f'ResNet-{6 * n + 2}')
    return model


def get_data_generators(model_name):
    if model_name in ['VGG16', 'VGG19', 'ResNet']:
        if model_name == 'VGG16':
            preproc = tf.keras.applications.vgg16.preprocess_input
        elif model_name == 'VGG19':
            preproc = tf.keras.applications.vgg19.preprocess_input
        else:
            preproc = tf.keras.applications.resnet50.preprocess_input
        rescale = None
    else:
        preproc = None
        rescale = 1. / 255

    all_files = []
    for f in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, f)
        if os.path.isfile(full_path) and f.lower().endswith(IMAGE_EXTENSIONS):
            all_files.append(f)

    bird_files = [f for f in all_files if 'bird' in f.lower()]
    drone_files = [f for f in all_files if 'drone' in f.lower()]

    def split_class(file_list):
        train, test = train_test_split(file_list, test_size=0.2, random_state=42)
        return train, test

    bird_train, bird_test = split_class(bird_files)
    drone_train, drone_test = split_class(drone_files)

    train_files = bird_train + drone_train
    train_labels = ['bird'] * len(bird_train) + ['drone'] * len(drone_train)

    test_files = bird_test + drone_test
    test_labels = ['bird'] * len(bird_test) + ['drone'] * len(drone_test)

    valid_files = []
    valid_labels = []
    for class_name, label_str in [('bird', 'bird'), ('drone', 'drone')]:
        class_dir = os.path.join(VALID_DIR, class_name)
        if os.path.isdir(class_dir):
            for f in os.listdir(class_dir):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    valid_files.append(os.path.join('valid', class_name, f))
                    valid_labels.append(label_str)

    train_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preproc,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    test_valid_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preproc
    )

    train_df = pd.DataFrame({'filename': train_files, 'class': train_labels})
    test_df = pd.DataFrame({'filename': test_files, 'class': test_labels})
    valid_df = pd.DataFrame({'filename': valid_files, 'class': valid_labels})

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=DATA_DIR,
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True
    )

    test_gen = test_valid_datagen.flow_from_dataframe(
        test_df,
        directory=DATA_DIR,
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    valid_gen = test_valid_datagen.flow_from_dataframe(
        valid_df,
        directory=DATA_DIR,
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    return train_gen, valid_gen, test_gen
