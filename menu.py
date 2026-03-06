import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt

from main import *

MODELS_DIR = 'models'
METRICS_DIR = 'metrics'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_epochs():
    while True:
        epochs = input('\nВведите количество эпох для обучения: ').strip()
        try:
            epochs = int(epochs)
            if epochs <= 0:
                print('Количество эпох должно быть положительным числом.')
            else:
                return epochs
        except ValueError:
            print('Некорректный ввод.')


def model_info(base):
    metric_path = os.path.join(METRICS_DIR, base.strip('.keras') + '_metrics.txt')
    if os.path.exists(metric_path):
        with (open(metric_path, 'r', encoding='utf-8') as f):
            metrics = list(map(lambda x: x.strip('\n'), f.readlines()))
            print(metrics[1])
            print(metrics[4] + ' | ' + metrics[5])
            print(metrics[2] + ' | ' + metrics[3])


def train_model(model_name):
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

    epochs = get_epochs()

    train_gen, valid_gen, test_gen = get_data_generators(model_name)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    print(f'\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'{model_name}_{timestamp}.keras'
    model_path = os.path.join(MODELS_DIR, model_filename)
    model.save(model_path)
    print(f'Модель сохранена: {model_path}')

    class_indices = train_gen.class_indices
    indices_to_class = {v: k for k, v in class_indices.items()}
    mapping_path = os.path.join(METRICS_DIR, f'{model_name}_{timestamp}_classes.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(indices_to_class, f, indent=4, ensure_ascii=False)

    metrics_path = os.path.join(METRICS_DIR, f'{model_name}_{timestamp}_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f'Model name: {model_name}\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Test loss: {test_loss:.4f}\n')
        f.write(f'Test accuracy: {test_acc:.4f}\n')
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        f.write(f'Train accuracy: {final_train_acc:.4f}\n')
        f.write(f'Val accuracy: {final_val_acc:.4f}\n')

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='val_acc', linewidth=2)
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
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
    input("Нажмите Enter, чтобы продолжить...")


def predict_with_model():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]
    if not model_files:
        print('Нет сохранённых моделей. Сначала обучите модель.')
        input("Нажмите Enter, чтобы продолжить...")
        return

    while True:
        print('=' * 40 + '\n' + ' ' * 12 + 'ДОСТУПНЫЕ МОДЕЛИ' + ' ' * 12 + '\n' + '=' * 40)
        for i, fname in enumerate(model_files, 1):
            print(f'{i}. {fname}')
        print('0. Назад')

        choice = input('Выберите номер модели: ').strip()
        if choice == '0':
            return
        idx = int(choice) - 1
        if idx < 0 or idx >= len(model_files):
            clear_console()
            print('Некорректный ввод.')
            continue
        break

    try:
        model_path = os.path.join(MODELS_DIR, model_files[idx])
        model = tf.keras.models.load_model(model_path)
        print(f'Модель {model_files[idx].strip('.keras')} успешно загружена.')
    except Exception as e:
        print(f'Ошибка загрузки модели: {e}')
        input("Нажмите Enter, чтобы продолжить...")
        return

    base = os.path.basename(model_path)
    model_name = base.split('_')[0]

    model_info(base)

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

    input("Нажмите Enter, чтобы продолжить...")
    clear_console()
    inf = True
    while True:
        if inf:
            print('=' * 40 + '\n' + ' ' * 11 + 'АНАЛИЗ ИЗОБРАЖЕНИЯ' + ' ' * 11 + '\n' + '=' * 40)
            print(f'Модель: {base.split('_')[0]}')
            model_info(base)
            print('(Для выхода в главное меню введите "0" вместо пути)')

            show_files = input('Показать список изображений в текущей папке? (y/n): ').strip().lower()
            if show_files == 'y':
                files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                if files:
                    print('Найденные изображения в текущей папке:')
                    for f in files:
                        print(f'  {f}')
                else:
                    print('В текущей папке нет изображений.')
            inf = False

        img_path = input('\nВведите путь к изображению: ').strip().strip('"').strip("'")
        if img_path == '0':
            return

        if not img_path:
            print('Путь не введён. Попробуйте снова.')
            continue

        if not os.path.exists(img_path):
            print('Файл не найден. Попробуйте снова.')
            continue

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

            print(f'Результат: {class_name} (индекс {predicted_class})')
            print(f'Вероятности: {probabilities}')

        except Exception as e:
            print(f'Ошибка при обработке изображения: {e}')


def submenu_train():
    while True:
        print('=' * 40 + '\n' + ' ' * 7 + 'ВЫБОР МОДЕЛИ ДЛЯ ОБУЧЕНИЯ' + ' ' * 8 + '\n' + '=' * 40)
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
            return
        else:
            clear_console()
            print('Некорректный ввод.')
            continue
        clear_console()


def main_menu():
    clear_console()
    while True:
        print('=' * 40 + '\n' + ' ' * 14 + 'ГЛАВНОЕ МЕНЮ' + ' ' * 14 + '\n' + '=' * 40)
        print('1. Обучение модели')
        print('2. Определить предмет на картинке')
        print('0. Выйти')
        choice = input('Выберите пункт: ').strip()

        clear_console()
        if choice == '1':
            submenu_train()
        elif choice == '2':
            predict_with_model()
        elif choice == '0':
            return
        else:
            print('Некорректный ввод. Повторите.')
            continue
        clear_console()


if __name__ == '__main__':
    main_menu()
