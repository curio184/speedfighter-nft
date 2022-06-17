import datetime as dt
from typing import List

import cv2

try:
    # Desktop Only
    from speedfighter.digit_recognition.image_classifier import ImageClassifier
    from speedfighter.digit_recognition.tfmodel_converter import \
        TFModelConverter
except:
    pass
from speedfighter.digit_recognition.image_classifier_lite import \
    ImageClassifierLite
from speedfighter.digit_recognition.seven_seg_splitter import SevenSegSplitter
from speedfighter.digit_recognition.training_dataset_creator import \
    TrainingDatasetCreator
from speedfighter.digit_recognition.video_camera import VideoCamera
from speedfighter.utils.directory import Directory
from speedfighter.utils.image_editor import ImageEditor


class DigitRecognitionUsecase:

    @staticmethod
    def collect_training_data():
        """
        ビデオカメラで学習対象を録画し、学習素材を録画する
        """

        # 画像保存ディレクトリ
        save_path = "./assets/rawdata/{:%y%m%d_%H%M%S}/".format(dt.datetime.now())

        # ビデオカメラで学習対象を録画し、学習素材を収集する
        camera = VideoCamera()
        camera.record(save_path)

    @staticmethod
    def play_training_data():
        """
        録画を再生する
        """

        # 録画を再生する
        camera = VideoCamera()
        camera.play("./assets/rawdata/220604_174800")

    @staticmethod
    def create_training_dataset():
        """
        学習素材をカテゴリに分類し、データセットを作成する
        """

        # カテゴリを定義する
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "blank", "no_signal"]

        # 素材画像のディレクトリパス
        rawdata_dir_path = "./assets/rawdata/220531_181635/"

        # データセットを作成する
        createor = TrainingDatasetCreator()
        createor.create_dataset(class_names, rawdata_dir_path)

    @staticmethod
    def build_and_train_model():
        """
        7-Seg画像のカテゴリを学習する
        """

        # モデルの配置場所
        dataset_dir_path = "./assets/dataset"
        model_h5_path = "./assets/keras_cnn_7segment_digits_28_28_gray_model.h5"
        class_names_json_path = "./assets/keras_cnn_7segment_digits_28_28_gray_class_names.json"

        # 入力層の形状(画像の横幅/高さ/チャンネル)
        input_image_width: int = 28     # 画像を指定した形状に変換して学習する
        input_image_height: int = 28    # 画像を指定した形状に変換して学習する
        input_image_channel: int = 1    # モノクロで学習する場合「1」、カラーで学習する場合「3」

        # カテゴリ名
        class_names = Directory.get_directory_names(dataset_dir_path)

        # 出力層の次元数(画像のカテゴリ数)
        output_dimensions = len(class_names)

        # 画像分類器
        classifier = ImageClassifier()

        # モデルを構築する
        model = classifier.build_model(
            input_image_width,
            input_image_height,
            input_image_channel,
            output_dimensions
        )

        # データセットを構築する
        dataset = classifier.build_dataset(
            dataset_dir_path,
            input_image_width,
            input_image_height,
            input_image_channel
        )

        # モデルを学習する
        classifier.train(model, dataset, False)

        # モデルを保存する
        classifier.save_model(model, model_h5_path)

        # カテゴリ名を保存する
        classifier.save_class_names(class_names, class_names_json_path)

    @staticmethod
    def predict_image_category():
        """
        7-Seg画像のカテゴリを推測する(Tensor Flow版)
        """

        # モデルの配置場所
        model_h5_path = "./assets/keras_cnn_7segment_digits_28_28_gray_model.h5"
        class_names_json_path = "./assets/keras_cnn_7segment_digits_28_28_gray_class_names.json"

        # 素材画像のファイルパス
        file_paths = Directory.get_files("./assets/rawdata/220604_174800/*.jpg")

        # 画像分類器
        classifier = ImageClassifier()
        model = classifier.load_model(model_h5_path)
        class_names = classifier.load_class_names(class_names_json_path)

        for file_path in file_paths:

            # 7-Seg画像を読み込む
            mat_gbr = ImageEditor.load_as_mat(file_path)

            # 数字ごとの画像分割する
            splitter = SevenSegSplitter()
            mat_digits = splitter.split_into_digits(mat_gbr)

            # カテゴリを推測する
            numbers: List[str] = []
            for mat_digit in mat_digits:
                pre_number, pre_accuracy = classifier.predict(model, class_names, mat_digit)
                numbers.append(pre_number)

            # 推測結果を表示する
            mat_gbr = splitter.draw_bounding_box(mat_gbr, numbers)

            # フレームを表示する
            cv2.imshow("Capture", mat_gbr)

            # 33msec待機する(30fps)
            key_press = cv2.waitKey(33)

            # ESC押下でプログラムを終了する
            if key_press == 27:
                break

    @staticmethod
    def convert_tf_to_tflite():
        """
        Keras H5形式のモデルをtflite形式のモデルに変換する
        """

        # Keras H5形式のモデルのパス
        model_h5_path = "./assets/keras_cnn_7segment_digits_28_28_gray_model.h5"

        # tflite形式のモデルのパス
        model_tflite_path = "./assets/keras_cnn_7segment_digits_28_28_gray_model.tflite"

        # モデルを変換する
        TFModelConverter.kerash5_to_tflite(model_h5_path, model_tflite_path)

    @staticmethod
    def predict_image_category_tflite():
        """
        7-Seg画像のカテゴリを推測する(Tensor Flow Lite版)
        """

        # モデルの配置場所
        model_tflite_path = "./assets/keras_cnn_7segment_digits_28_28_gray_model.tflite"
        class_names_json_path = "./assets/keras_cnn_7segment_digits_28_28_gray_class_names.json"

        # 素材画像のファイルパス
        file_paths = Directory.get_files("./assets/rawdata/220604_174800/*.jpg")

        # 画像分類器
        classifier = ImageClassifierLite()
        interpreter = classifier.setup_interpreter(model_tflite_path)
        class_names = classifier.load_class_names(class_names_json_path)

        for file_path in file_paths:

            # 7-Seg画像を読み込む
            mat_gbr = ImageEditor.load_as_mat(file_path)

            # 数字ごとの画像分割する
            splitter = SevenSegSplitter()
            mat_digits = splitter.split_into_digits(mat_gbr)

            # カテゴリを推測する
            numbers: List[str] = []
            for mat_digit in mat_digits:
                pre_number, pre_accuracy = classifier.predict(interpreter, class_names, mat_digit)
                numbers.append(pre_number)

            # 推測結果を表示する
            mat_gbr = splitter.draw_bounding_box(mat_gbr, numbers)

            # フレームを表示する
            cv2.imshow("Capture", mat_gbr)

            # 33msec待機する(30fps)
            key_press = cv2.waitKey(33)

            # ESC押下でプログラムを終了する
            if key_press == 27:
                break


if __name__ == "__main__":

    # 1-1. カメラで対象を録画し、教師データを収集する(Raspberry Piで実行)
    DigitRecognitionUsecase.collect_training_data()

    # 1-2. 録画を再生し、教師データを確認する(PCで実行)
    # DigitRecognitionUsecase.play_training_data()

    # 1-3. 教師データを分類し、データセットを作成する(PCで実行)
    # DigitRecognitionUsecase.create_training_dataset()

    # 1-4. データセットでモデルを訓練し、画像のカテゴリを覚えさせる(PCで実行)
    # DigitRecognitionUsecase.build_and_train_model()

    # 1-5. TFモデルに画像を入力し、学習結果を確認する(PCで実行)
    # DigitRecognitionUsecase.predict_image_category()

    # 1-6. TF形式のモデルをTF LITE形式のモデルに変換する(PCで実行)
    # DigitRecognitionUsecase.convert_tf_to_tflite()

    # 1-7. TF Liteモデルに画像を入力し、学習結果を確認する(Raspberry Piで実行)
    # DigitRecognitionUsecase.predict_image_category_tflite()
