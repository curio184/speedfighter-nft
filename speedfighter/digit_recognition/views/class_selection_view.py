from typing import List

import matplotlib.pyplot as plt
from cv2 import Mat
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.image_editor import ImageEditor


class ClassSelectionView(AppBase):
    """
    画像のカテゴリを選択する画面
    """

    def __init__(self, default_input_text: str = ""):
        super().__init__()

        self._ax1_text = None

        # 入力されたキー
        self._input_text: List[str] = list(default_input_text)
        self._input_text_max_len = 3
        self._input_text_min_len = 3

        # 入力受付するキー
        self._acceptable_keys = list("0123456789qwertyuiopasdfghjklzxcvbnm_.-")

        # 入力受付するキーをショートカット割り当てから除外
        for param_key in plt.rcParams:
            if "keymap." in param_key:
                for acceptable_key in self._acceptable_keys:
                    if acceptable_key in plt.rcParams[param_key]:
                        self._logger.info(
                            "disable keymap. {}={}, {}".format(param_key, plt.rcParams[param_key], acceptable_key)
                        )
                        plt.rcParams[param_key].remove(acceptable_key)

    def show(self, mat_bgr: Mat, file_name: str, class_names: List[str]) -> List[str]:

        fig = plt.figure(figsize=(12, 6))
        fig.canvas.set_window_title("ClassSelection")

        ##############################
        # 画像と凡例をプロット
        ##############################

        # 1行、1列レイアウトの1番目
        ax1 = fig.add_subplot(1, 1, 1)

        # タイトル
        ax1.set_title(file_name, loc="right")

        # ラベル
        ax1.set_ylabel("height")
        ax1.set_xlabel("width")

        # 画像
        ax1.imshow(ImageEditor.bgr_to_rgb(mat_bgr))

        # 凡例
        labels = ["Select a class by key and press enter."]
        labels = labels + list(map(lambda x: "[{}]: {}".format(x[0], x), class_names))
        labels = labels + ["[backspace]: backspace", "[enter]: enter"]
        for label in labels:
            ax1.plot([], [], marker="s", label=label, linestyle="None")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left',)

        # キーイベントをバインド
        fig.canvas.mpl_connect('key_press_event', self._key_press)
        self._ax1_text = ax1.text(0, -5, "".join(self._input_text), fontsize=24)

        plt.tight_layout()
        plt.show(block=True)

        return self._input_text

    def _key_press(self, event):

        # enter, escapeキーの場合
        if event.key in ["enter", "escape"]:
            if len(self._input_text) < self._input_text_min_len:
                return
            if len(self._input_text) > self._input_text_max_len:
                return
            plt.close()
            return

        # 入力受付するキーの場合
        if event.key in self._acceptable_keys:
            self._input_text.append(event.key)

        # backspaceキーの場合
        if "backspace" in event.key:
            if len(self._input_text) >= 1:
                self._input_text = self._input_text[0:-1]

        # 入力を反映する
        self._ax1_text.set_text("".join(self._input_text))
        plt.draw()
