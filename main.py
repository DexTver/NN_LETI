import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19

# Константы
BATCH_SIZE = 32
EPOCHS = 3
IMG_SIZE = 224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
NUM_CLASSES = 2
TRAIN_DIR = 'dataset/train'
VALID_DIR = 'dataset/valid'
TEST_DIR = 'dataset/test'


# === 1. НАША АРХИТЕКТУРА (упрощённая, без циклов и параметров) ===
def create_my_model(input_shape, num_classes):
    model = models.Sequential(name='MyCNN')
    model.add(layers.Input(shape=input_shape))

    # Блок 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))

    # Блок 2
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))

    # Блок 3
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    # без пулинга после третьего слоя

    # Классификатор
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))

    return model


# === 2. ALEXNET (без изменений) ===
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

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))

    return model


# === 3. VGG16 ===
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


# === 4. VGG19 ===
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


# === ЗАГРУЗКА ДАННЫХ ===
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


# === ОБУЧЕНИЕ ===
models_dict = {
    'MyCNN': create_my_model(INPUT_SHAPE, NUM_CLASSES),
    # 'AlexNet': create_alexnet(INPUT_SHAPE, NUM_CLASSES),
    # 'VGG16': create_vgg16(INPUT_SHAPE, NUM_CLASSES, trainable_base=False),
    # 'VGG19': create_vgg19(INPUT_SHAPE, NUM_CLASSES, trainable_base=False)
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


# === ГРАФИКИ (все с диапазоном 0-1 по Y) ===
# 1. Сравнение всех моделей
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
for model_name in models_dict:
    if 'val_accuracy' in history_dict[model_name]:
        plt.plot(history_dict[model_name]['val_accuracy'], label=model_name, linewidth=2)
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # диапазон 0-1
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
for model_name in models_dict:
    if 'val_loss' in history_dict[model_name]:
        plt.plot(history_dict[model_name]['val_loss'], label=model_name, linewidth=2)
plt.title('Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 1)  # диапазон 0-1
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()


# 2. Графики для каждой модели отдельно
for model_name, hist in history_dict.items():
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(hist['accuracy'], label='train_acc', linewidth=2)
    plt.plot(hist['val_accuracy'], label='val_acc', linewidth=2)
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # диапазон 0-1
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(hist['loss'], label='train_loss', linewidth=2)
    plt.plot(hist['val_loss'], label='val_loss', linewidth=2)
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)  # диапазон 0-1
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png', dpi=150)
    plt.show()


# === ИТОГОВЫЕ МЕТРИКИ ===
print('\n' + '='*50)
print('ИТОГОВЫЕ ТЕСТОВЫЕ МЕТРИКИ')
print('='*50)
for model_name, metrics in test_metrics.items():
    print(f"{model_name:10s} - loss: {metrics['loss']:.4f} - accuracy: {metrics['accuracy']:.4f}")