import os
import json
import datetime
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
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

# Папки для сохранения
MODELS_DIR = 'models'
METRICS_DIR = 'metrics'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Скрыть главное окно tkinter
root = tk.Tk()
root.withdraw()


# === МОДЕЛИ (без изменений) ===
def create_my_model(input_shape, num_classes):
    model = models.Sequential(name='MyCNN')
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
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


# === ГЕНЕРАТОРЫ ДАННЫХ ===
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


# === ОБУЧЕНИЕ ОДНОЙ МОДЕЛИ С СОХРАНЕНИЕМ ===
def train_model(model_name):
    print(f'\n{"="*50}\nОбучение модели {model_name}\n{"="*50}')

    # Создание модели
    if model_name == 'MyCNN':
        model = create_my_model(INPUT_SHAPE, NUM_CLASSES)
    elif model_name == 'AlexNet':
        model = create_alexnet(INPUT_SHAPE, NUM_CLASSES)
    elif model_name == 'VGG16':
        model = create_vgg16(INPUT_SHAPE, NUM_CLASSES, trainable_base=False)
    elif model_name == 'VGG19':
        model = create_vgg19(INPUT_SHAPE, NUM_CLASSES, trainable_base=False)
    else:
        print('Неизвестная модель')
        return

    # Генераторы
    train_gen, valid_gen, test_gen = get_data_generators(model_name)

    # Компиляция
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    # Обучение
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=valid_gen,
        verbose=1
    )

    # Оценка на тесте
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f'\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

    # Сохранение модели
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'{model_name}_{timestamp}.keras'
    model_path = os.path.join(MODELS_DIR, model_filename)
    model.save(model_path)
    print(f'Модель сохранена: {model_path}')

    # Сохранение class_indices (mapping классов)
    class_indices = train_gen.class_indices  # {'class0': 0, 'class1': 1}
    indices_to_class = {v: k for k, v in class_indices.items()}
    mapping_path = os.path.join(METRICS_DIR, f'{model_name}_{timestamp}_classes.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(indices_to_class, f, indent=4, ensure_ascii=False)

    # Сохранение метрик (текстовый файл)
    metrics_path = os.path.join(METRICS_DIR, f'{model_name}_{timestamp}_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f'Test loss: {test_loss:.4f}\n')
        f.write(f'Test accuracy: {test_acc:.4f}\n')
        # Можно также записать последние значения accuracy/loss
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        f.write(f'Final train accuracy: {final_train_acc:.4f}\n')
        f.write(f'Final val accuracy: {final_val_acc:.4f}\n')

    # Построение и сохранение графиков для данной модели
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='val_acc', linewidth=2)
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='val_loss', linewidth=2)
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(METRICS_DIR, f'{model_name}_{timestamp}_training.png')
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f'Графики сохранены: {plot_path}')


# === ПРЕДСКАЗАНИЕ ДЛЯ ОДНОГО ИЗОБРАЖЕНИЯ ===
import os

def predict_with_model():
    # Получаем список сохранённых моделей
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
    if not model_files:
        print('Нет сохранённых моделей. Сначала обучите модель.')
        return

    print('\nДоступные модели:')
    for i, fname in enumerate(model_files, 1):
        print(f'{i}. {fname}')
    print('0. Назад')

    choice = input('Выберите номер модели: ').strip()
    if choice == '0':
        return
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(model_files):
            raise ValueError
        model_path = os.path.join(MODELS_DIR, model_files[idx])
    except:
        print('Некорректный ввод.')
        return

    # Загружаем модель
    try:
        model = tf.keras.models.load_model(model_path)
        print('Модель успешно загружена.')
    except Exception as e:
        print(f'Ошибка загрузки модели: {e}')
        return

    # Определяем имя модели из имени файла
    base = os.path.basename(model_path)
    model_name = base.split('_')[0]

    # Загрузка маппинга классов
    timestamp_part = '_'.join(base.split('_')[1:]).replace('.keras', '')
    mapping_filename = f'{model_name}_{timestamp_part}_classes.json'
    mapping_path = os.path.join(METRICS_DIR, mapping_filename)
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            indices_to_class = json.load(f)
        indices_to_class = {int(k): v for k, v in indices_to_class.items()}
        print('Маппинг классов загружен.')
    else:
        indices_to_class = {0: 'класс 0', 1: 'класс 1'}
        print('Внимание: файл с именами классов не найден, используются индексы.')

    # Цикл для множественных предсказаний
    while True:
        print('\n--- Анализ изображения ---')
        print('(Для выхода в главное меню введите "0" или "q" вместо пути)')

        # Опция: показать файлы в текущей папке
        show_files = input('Показать список изображений в текущей папке? (y/n): ').strip().lower()
        if show_files == 'y':
            files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if files:
                print('Найденные изображения в текущей папке:')
                for f in files:
                    print(f'  {f}')
            else:
                print('В текущей папке нет изображений.')

        img_path = input('Введите путь к изображению: ').strip().strip('"').strip("'")
        if img_path in ('0', 'q'):
            print('Возврат в главное меню.')
            break

        if not img_path:
            print('Путь не введён. Попробуйте снова.')
            continue

        if not os.path.exists(img_path):
            print('Файл не найден. Попробуйте снова.')
            continue

        # Загрузка и предобработка изображения
        try:
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            if model_name in ['VGG16', 'VGG19']:
                if model_name == 'VGG16':
                    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
                else:
                    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
            else:
                img_array = img_array / 255.0

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            class_name = indices_to_class.get(predicted_class, f'неизвестный класс {predicted_class}')
            probabilities = tf.nn.softmax(predictions[0]).numpy()

            print(f'\nРезультат: {class_name} (индекс {predicted_class})')
            print(f'Вероятности: {probabilities}')

        except Exception as e:
            print(f'Ошибка при обработке изображения: {e}')
            continue

        # Спросить, хочет ли пользователь продолжить с другим изображением
        again = input('\nХотите проверить другое изображение этой же моделью? (y/n): ').strip().lower()
        if again != 'y':
            print('Возврат в главное меню.')
            break


# === МЕНЮ ===
def main_menu():
    while True:
        print('\n' + '='*40)
        print('ГЛАВНОЕ МЕНЮ')
        print('='*40)
        print('1. Обучение модели')
        print('2. Определить предмет на картинке')
        print('0. Выйти')
        choice = input('Выберите пункт: ').strip()

        if choice == '1':
            submenu_train()
        elif choice == '2':
            predict_with_model()
        elif choice == '0':
            break
        else:
            print('Некорректный ввод. Повторите.')

def submenu_train():
    while True:
        print('\n' + '-'*30)
        print('ВЫБОР МОДЕЛИ ДЛЯ ОБУЧЕНИЯ')
        print('-'*30)
        print('1. MyCNN')
        print('2. AlexNet')
        print('3. VGG16')
        print('4. VGG19')
        print('0. Назад')
        choice = input('Выберите модель: ').strip()

        if choice == '1':
            train_model('MyCNN')
        elif choice == '2':
            train_model('AlexNet')
        elif choice == '3':
            train_model('VGG16')
        elif choice == '4':
            train_model('VGG19')
        elif choice == '0':
            break
        else:
            print('Некорректный ввод.')


if __name__ == '__main__':
    main_menu()