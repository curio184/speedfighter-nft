import tensorflow as tf
from keras import models


class TFModelConverter:
    """
    TensorFlowのモデルコンバーター
    """

    @staticmethod
    def kerash5_to_tflite(model_h5_path: str, model_tflite_path: str):
        """
        Keras H5形式のモデルをtflite形式のモデルに変換する

        Parameters
        ----------
        model_h5_path : str
            Keras H5形式のモデルのパス
        model_tflite_path : str
            tflite形式のモデルのパス
        """

        # Keras H5形式のモデルを読み込む
        model = models.load_model(model_h5_path)

        # tflite形式のモデルに変換
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        open(model_tflite_path, "wb").write(tflite_model)



