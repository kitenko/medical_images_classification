import os
import json

from tensorflow import keras

from config import LOGS_DIR, SAVE_MODELS


class LogCallback(keras.callbacks.Callback):
    """
    Логирование всех метрик обучения в json файл и сохранение модели каждую эпоху.
    Использовать в виде контекст-менеджера.

    :param model_save_path: путь к папке, в которую сохранять модели.
    :param logs_save_path: путь к папке, в которую сохранять логи.
    """
    def __init__(self, model_save_path: str = SAVE_MODELS, logs_save_path: str = LOGS_DIR):
        super().__init__()
        self.model_save_path = model_save_path
        self.log_file = os.path.join(logs_save_path, 'train_logs.json')
        self.logs = {'epochs': []}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_logs()
        self.model.save(os.path.join(self.model_save_path, 'last.h5'))

    def save_logs(self):
        with open(self.log_file, 'w') as file:
            json.dump(self.logs, file, indent=4)

    def on_epoch_end(self, epoch, logs=None):
        text = ['epoch: {:03d}'.format(epoch + 1)]
        for key, value in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(float(value))
            text.append('{}: {:.04f}'.format(key, float(value)))
        self.logs['epochs'].append('; '.join(text))
        self.save_logs()
        # self.model.save(os.path.join(self.model_save_path, '{:03d}_epoch.h5'.format(1 + epoch)))
