import matplotlib.pyplot as plt
from speedfighter.utils.app_base import AppBase


class TrainHistoryView(AppBase):
    """
    モデルの学習履歴を表示する画面
    """

    def __init__(self):
        super().__init__()

    def show(self, history):
        """
        モデルの学習履歴を表示する
        """

        self._logger.info(history.history.keys())
        accuracy = history.history["accuracy"]
        val_accuracy = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(accuracy) + 1)

        fig = plt.figure()
        fig.canvas.set_window_title("TrainHistory")

        ##############################
        # 精度の履歴をプロット
        ##############################

        # 1行、2列レイアウトの1番目
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Training and validation accuracy")
        ax1.plot(epochs, accuracy, "bo", label="Training acc")
        ax1.plot(epochs, val_accuracy, "b", label="Validation acc")
        ax1.legend()
        ax1.grid()

        ##############################
        # 損失の履歴をプロット
        ##############################

        # 1行、2列レイアウトの2番目
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("Training and validation loss")
        ax2.plot(epochs, loss, "bo", label="Training acc")
        ax2.plot(epochs, val_loss, "b", label="Validation loss")
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.show()
