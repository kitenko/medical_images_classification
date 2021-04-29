import json

import matplotlib.pyplot as plt


class Graf():
    def __init__(self, path_for_json_file_logs: str, graf_loss: bool = True, graf_accuracy: bool = True,
                 graf_recall: bool = True, graf_precision: bool = True, graf_f1score: bool = True) -> None:
        """
        Depending on the entered parameters, the function will plot different graphs.

        :param path_for_json_file_logs: This is the path for the json file with logs.
        :param graf_loss: To plot or not to plot the loss.
        :param graf_accuracy: To plot or not to plot the accuracy.
        :param graf_recall: To plot or not to plot the recall.
        :param graf_precision: To plot or not to plot the precision.
        :param graf_f1score: To plot or not to plot the f1score.
        """
        self.path_for_json_file_logs = path_for_json_file_logs
        self.graf_loss = graf_loss
        self.graf_accuracy = graf_accuracy
        self.graf_recall = graf_recall
        self.graf_precision = graf_precision
        self.graf_f1score = graf_f1score

    def plot_graphics(self) -> None:
        """
        This function plots.

        :return:
        """
        with open(self.path_for_json_file_logs) as json_file:
            data = json.load(json_file)

            # parameters for plots
            acc = data['accuracy']
            val_acc = data['val_accuracy']
            loss = data['loss']
            val_loss = data['val_loss']
            precision = data['precision']
            val_precision = data['val_precision']
            recall = data['recall']
            val_recall = data['val_recall']
            f1_score = data['F1_score']
            val_f1_score = data['val_F1_score']
            epochs_range = range(len(data['epochs']))

            plt.figure(figsize=(20, 20))

            if self.graf_accuracy:
                plt.subplot(1, 5, 1)
                plt.plot(epochs_range, acc, label='Training Accuracy')
                plt.plot(epochs_range, val_acc, label='Validation Accuracy')
                plt.legend(loc='lower right')
                plt.title('Training and Validation Accuracy')

            if self.graf_loss:
                plt.subplot(1, 5, 2)
                plt.plot(epochs_range, loss, label='Training Loss')
                plt.plot(epochs_range, val_loss, label='Validation Loss')
                plt.legend(loc='upper right')
                plt.title('Training and Validation Loss')

            if self.graf_precision:
                plt.subplot(1, 5, 3)
                plt.plot(epochs_range, precision, label='Training Precision')
                plt.plot(epochs_range, val_precision, label='Validation Precision')
                plt.legend(loc='upper right')
                plt.title('Training and Validation Precision')

            if self.graf_recall:
                plt.subplot(1, 5, 4)
                plt.plot(epochs_range, recall, label='Training Recall')
                plt.plot(epochs_range, val_recall, label='Validation Recall')
                plt.legend(loc='upper right')
                plt.title('Training and Validation Recall')

            if self.graf_f1score:
                plt.subplot(1, 5, 5)
                plt.plot(epochs_range, f1_score, label='Training F1_Score')
                plt.plot(epochs_range, val_f1_score, label='Validation F1_Score')
                plt.legend(loc='upper right')
                plt.title('Training and Validation F1_Score')

            plt.show()
