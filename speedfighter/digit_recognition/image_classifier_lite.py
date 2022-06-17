import codecs
import json
import time
from typing import List, Tuple

import numpy as np
from cv2 import Mat

try:
    # for raspberry pi
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # for windows
    from tensorflow.lite.python.interpreter import Interpreter

from speedfighter.digit_recognition.views.prediction_result_view import \
    PredictionResultView
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.image_editor import ImageEditor


class ImageClassifierLite(AppBase):
    """
    画像分類器(TensorFlow Lite版)

    Raspberry Piなどの小型コンピューターでTensorFlowを実行するには、
    エッジデバイスに最適化されたTensorFlow Liteを利用する。
    TensorFlowで学習したモデルをLite用に変換することで利用できる。
    ただし、TensorFlow Lite単体でモデルの構築・学習はできない。
    """

    def __init__(self, verbose: bool = False):
        super().__init__()
        self._verbose = verbose

    def setup_interpreter(self, model_tflite_path: str) -> Interpreter:
        """
        インタープリターをセットアップする

        Parameters
        ----------
        model_tflite_path : str
            モデルのファイルパス(Lite形式, model.tflite)

        Returns
        -------
        Interpreter
            インタープリター
        """

        # モデルを読み込む
        interpreter = Interpreter(model_path=model_tflite_path, num_threads=4)
        interpreter.allocate_tensors()

        return interpreter

    def load_class_names(self, class_names_json_path: str) -> List[str]:
        """
        カテゴリ名(class_names.json)を読み込む

        Parameters
        ----------
        class_names_json_path : str
            カテゴリ名のファイルパス

        Returns
        -------
        List[str]
            カテゴリ名の一覧
        """
        # カテゴリ名を読み込む
        with codecs.open(class_names_json_path, "r", "utf8") as f:
            return json.load(f)

    def predict(self, interpreter: Interpreter, class_names: List[str], mat_bgr: Mat) -> Tuple[str, int]:
        """
        学習済みのモデルを利用し、画像のカテゴリを推測します。

        Parameters
        ----------
        model: Interpreter
            インタープリター
        class_names : List[str]
            カテゴリ名の一覧
        mat_bgr : Mat
            Mat形式の3チャンネルカラー画像

        Returns
        -------
        Tuple[str, int]
            (カテゴリ名, 精度0~100)
        """

        # 入力層の形状(画像の横幅/高さ/チャンネル)
        input_details = interpreter.get_input_details()
        input_image_width = input_details[0]["shape"][1]
        input_image_height = input_details[0]["shape"][2]
        input_image_channel = input_details[0]["shape"][3]

        # 出力層の次元数(画像のカテゴリ数)
        output_details = interpreter.get_output_details()
        output_dimensions = output_details[0]["shape"][1]

        # 評価対象を入力形状に変換する
        eval_image = None
        if input_image_channel == 1:
            eval_image = ImageEditor.bgr_to_gray(mat_bgr)
        elif input_image_channel == 3:
            eval_image = ImageEditor.bgr_to_rgb(mat_bgr)
        eval_image = ImageEditor.resize(eval_image, input_image_width, input_image_height)
        eval_image = ImageEditor.mat_to_3d_array(eval_image)
        eval_image = ImageEditor.normalize(eval_image)

        # 画像のカテゴリを推測する
        interpreter.set_tensor(input_details[0]['index'], np.array([eval_image]))

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        prediction = predictions[0]

        # 推測結果を表示する
        if self._verbose:
            for idx, accuracy in enumerate(prediction):
                self._logger.info("{}の確率: {}%".format(class_names[idx], int(accuracy * 100)))
            pr_view = PredictionResultView()
            pr_view.show(mat_bgr, prediction, class_names)

        self._logger.info("{}: {}%, time: {:.3f}ms".format(
            class_names[prediction.argmax()],
            prediction[prediction.argmax()],
            ((stop_time - start_time) * 1000)
        ))

        return (class_names[prediction.argmax()], int(prediction[prediction.argmax()] * 100))
