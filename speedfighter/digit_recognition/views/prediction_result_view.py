from typing import List

import matplotlib.pyplot as plt
import numpy as np
from cv2 import Mat
from numpy.typing import NDArray
from speedfighter.utils.app_base import AppBase


class PredictionResultView(AppBase):
    """
    モデルの推測結果を表示する画面
    """

    def __init__(self):
        super().__init__()

    def show(self, mat: Mat, prediction: NDArray, class_names: List[str]):
        """
        モデルの学習履歴を表示する
        """

        ##############################
        # ウインドウ
        ##############################

        fig = plt.figure()
        fig.canvas.set_window_title("PredictionResult")

        ##############################
        # 推測画像
        ##############################

        # 1行、2列レイアウトの1番目
        ax1 = fig.add_subplot(1, 2, 1)

        # ax1.imshow(mat, cmap=plt.cm.binary)
        ax1.imshow(mat)

        xlabel = "{} {:2.0f}%".format(class_names[prediction.argmax()], 100*np.max(prediction))
        ax1.set_xlabel(xlabel, color="blue")

        # ax1.grid(False)
        # ax1.set_xticks([])
        # ax1.set_yticks([])

        ##############################
        # 推測結果
        ##############################

        # 1行、2列レイアウトの1番目
        ax2 = fig.add_subplot(1, 2, 2)
        thisplot = ax2.bar(class_names, prediction, color="#777777")
        thisplot[prediction.argmax()].set_color('blue')
        ax2.set_xlabel("class_names")
        ax2.set_ylabel("accuracy")
        ax2.grid(False)
        # ax2.set_xticks([])                # X軸のメモリ
        ax2.set_yticks([0.0, 0.5, 1.0])     # Y軸のメモリ
        ax2.set_ylim([0, 1])                # Y軸のメモリ範囲

        plt.tight_layout()
        plt.show()
