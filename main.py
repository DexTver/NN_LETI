import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19


BATCH_SIZE = 32
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
NUM_CLASSES = 2
TRAIN_DIR = 'dataset/train'
VALID_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'


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


def get_data_generators(model_name):
    if model_name in ['VGG16', 'VGG19']:
        if model_name == 'VGG16':
            preproc = tf.keras.applications.vgg16.preprocess_input
        else:
            preproc = tf.keras.applications.vgg19.preprocess_input
        rescale = None
    else:
        preproc = None
        rescale = 1. / 255

    train_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preproc,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preproc
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True
    )

    valid_gen = test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    return train_gen, valid_gen, test_gen
