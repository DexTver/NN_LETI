import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19

BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
NUM_CLASSES = 2
TRAIN_DIR = 'dataset/train'
VALID_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'


def create_my_model(input_shape, num_classes, params=None):
    """
    params: словарь с настраиваемыми параметрами:
        - conv_blocks: список кортежей (filters, kernel_size, pool_size) для свёрточных блоков
        - dense_units: список чисел нейронов в полносвязных слоях
        - dropout_rate: вероятность dropout (если нужен)
    По умолчанию используется архитектура из презентации (3 сверточных слоя + 2 полносвязных)
    """
    if params is None:
        params = {
            'conv_blocks': [(32, (3,3), (2,2)), (64, (3,3), (2,2)), (64, (3,3), None)],
            'dense_units': [64],
            'dropout_rate': 0.0
        }

    model = models.Sequential(name='My_CNN')
    model.add(layers.Input(shape=input_shape))

    # Свёрточные блоки
    for i, (filters, kernel_size, pool_size) in enumerate(params['conv_blocks']):
        model.add(layers.Conv2D(filters, kernel_size, activation='relu', padding='same'))
        if pool_size is not None:
            model.add(layers.MaxPooling2D(pool_size))

    model.add(layers.Flatten())

    # Полносвязные слои
    for units in params['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
        if params['dropout_rate'] > 0:
            model.add(layers.Dropout(params['dropout_rate']))

    # Выходной слой
    model.add(layers.Dense(num_classes))

    return model


def create_alexnet(input_shape, num_classes):
    model = models.Sequential(name='AlexNet')
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(96, (11,11), strides=4, activation='relu', padding='valid'))
    model.add(layers.MaxPooling2D((3,3), strides=2))
    model.add(layers.Conv2D(256, (5,5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((3,3), strides=2))
    model.add(layers.Conv2D(384, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(384, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((3,3), strides=2))

    # Классификатор
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))

    return model


def create_vgg16(input_shape, num_classes, trainable_base=False):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = trainable_base   # заморозка базовых слоёв

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
        rescale = 1./255

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

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True
    )

    valid_generator = test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    return train_generator, valid_generator, test_generator


models_dict = {
    'MyCNN': create_my_model(INPUT_SHAPE, NUM_CLASSES),
    'AlexNet': create_alexnet(INPUT_SHAPE, NUM_CLASSES),
    'VGG16': create_vgg16(INPUT_SHAPE, NUM_CLASSES, trainable_base=False),
    'VGG19': create_vgg19(INPUT_SHAPE, NUM_CLASSES, trainable_base=False)
}
history_dict = {}
test_metrics = {}

for model_name, model in models_dict.items():
    print(f'\n{"="*50}\nОбучение модели {model_name}\n{"="*50}')

    train_gen, valid_gen, test_gen = get_data_generators(model_name)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.summary()
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=valid_gen,
        verbose=1
    )
    history_dict[model_name] = history.history
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    test_metrics[model_name] = {'loss': test_loss, 'accuracy': test_acc}
    print(f'\n{model_name} - Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
for model_name in models_dict:
    if 'val_accuracy' in history_dict[model_name]:
        plt.plot(history_dict[model_name]['val_accuracy'], label=model_name)
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
for model_name in models_dict:
    if 'val_loss' in history_dict[model_name]:
        plt.plot(history_dict[model_name]['val_loss'], label=model_name)
plt.title('Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

print('\n' + '='*50)
print('ИТОГОВЫЕ ТЕСТОВЫЕ МЕТРИКИ')
print('='*50)
for model_name, metrics in test_metrics.items():
    print(f"{model_name:10s} - loss: {metrics['loss']:.4f} - accuracy: {metrics['accuracy']:.4f}")

for model_name, hist in history_dict.items():
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist['accuracy'], label='train_acc')
    plt.plot(hist['val_accuracy'], label='val_acc')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist['loss'], label='train_loss')
    plt.plot(hist['val_loss'], label='val_loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png')
    plt.show()
