import codecs
import json
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cv2 import Mat
from keras import layers, models, optimizers
from keras.applications import VGG16
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from numpy.typing import NDArray
from sklearn import model_selection
from speedfighter.digit_recognition.views.prediction_result_view import \
    PredictionResultView
from speedfighter.digit_recognition.views.train_history_view import \
    TrainHistoryView
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.directory import Directory
from speedfighter.utils.image_editor import ImageEditor


class ImageClassifier(AppBase):
    """
    画像分類器(TensorFlow 2版)

    モデルの構築・学習、そして推測を実行する。
    学習したモデルはコンバーターを通じて、TensorFlow Liteで利用できる。
    """

    def __init__(self, verbose: bool = False):
        super().__init__()
        self._verbose = verbose

    def build_model(self, input_image_width: int, input_image_height: int, input_image_channel: int, output_dimensions: int) -> Sequential:
        """
        モデルを構築する

        Parameters
        ----------
        input_image_width : int
            入力層の形状(画像の横幅)
        input_image_height : int
            入力層の形状(画像の高さ)
        input_image_channel : int
            入力層の形状(画像のチャンネル)
            モノクロで学習する場合「1」、カラーで学習する場合「3」
        output_dimensions : int
            出力層の次元数(画像のカテゴリ数)

        Returns
        -------
        Sequential
            モデル
        """

        version = "cnn_v2"

        if version == "cnn_v1":

            # モデルを定義する
            model = models.Sequential()

            # レイヤーを定義する
            model.add(layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(input_image_width, input_image_height, input_image_channel)))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.Dense(output_dimensions, activation='sigmoid'))

            # モデルのレイヤーを出力する
            self._logger.info(model.summary())

            # モデルをコンパイルする
            model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

            return model

        elif version == "cnn_v2":

            # モデルを定義する
            model = Sequential()

            # レイヤーを定義する
            model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu',
                                    input_shape=(input_image_width, input_image_height, input_image_channel)))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Dropout(0.5))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.25))
            model.add(layers.Dense(output_dimensions, activation='softmax'))

            # モデルのレイヤーを出力する
            self._logger.info(model.summary())

            # モデルをコンパイルする
            model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

            return model

        elif version == "vgg16":

            # モデルを定義する
            model = models.Sequential()

            # レイヤーを定義する
            conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(
                input_image_width, input_image_height, input_image_channel))
            conv_base.trainable = False
            model.add(conv_base)
            model.add(layers.Flatten())
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dense(output_dimensions, activation='sigmoid'))

            # モデルのレイヤーを出力する
            self._logger.info(model.summary())

            # モデルをコンパイルする
            model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

            return model

    def save_model(self, model: Sequential, model_h5_path: str):
        """
        モデルを保存する
        ※モデルには、アーキテクチャ、重み値、およびcompile()情報が含まれる。
        ※モデルには、カテゴリ名が含まれないため、別途保存する必要がある。

        Parameters
        ----------
        model : Sequential
            モデル
        model_h5_path : str
            モデルのファイルパス(Keras H5形式, model.h5)
        """

        # モデルを保存する
        model.save(model_h5_path)

        # モデルのグラフ構造を画像で保存する
        # from keras.utils import plot_model
        # plot_model(model, to_file="./assets/model.png")

    def load_model(self, model_h5_path: str) -> Sequential:
        """
        モデルを読み込む

        Parameters
        ----------
        model_h5_path : str
            モデルのファイルパス(Keras H5形式, model.h5)

        Returns
        -------
        Sequential
            モデル
        """

        # モデルを読み込む
        model = models.load_model(model_h5_path)

        # モデルのレイヤーを出力する
        self._logger.info(model.summary())

        return model

    def save_class_names(self, class_names: List[str], class_names_json_path: str):
        """
        カテゴリ名(class_names.json)を保存する

        Parameters
        ----------
        class_names : List[str]
            カテゴリ名の一覧
        class_names_json_path : str
            カテゴリ名のファイルパス
        """

        # カテゴリ名を保存する
        with codecs.open(class_names_json_path, "w", "utf8") as f:
            json.dump(class_names, f, ensure_ascii=False)

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

    def save_weight(self, model: Sequential, weight_h5_path: str):
        """
        モデルの重みを保存する
        ※Keras H5形式のモデルは重みを含むため、通常は利用しなくてよい。

        Parameters
        ----------
        model : Sequential
            モデル
        weight_h5_path : str
            重みのファイルパス
        """
        # モデルの重みを保存する
        model.save_weights(weight_h5_path)

    def load_weight(self, model: Sequential, weight_h5_path: str):
        """
        モデルの重みを読み込む
        ※Keras H5形式のモデルは重みを含むため、通常は利用しなくてよい。

        Parameters
        ----------
        model : Sequential
            モデル
        weight_h5_path : str
            重みのファイルパス
        """
        # モデルの重みを読み込む
        model.load_weights(weight_h5_path)

    def build_dataset(self, dataset_dir_path: str, input_image_width: int, input_image_height: int, input_image_channel: int) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        データセットを構築する

        Parameters
        ----------
        dataset_dir_path : str
            データセットのディレクトリパス
        input_image_width : int
            入力層の形状(画像の横幅)
        input_image_height : int
            入力層の形状(画像の高さ)
        input_image_channel : int
            入力層の形状(画像のチャンネル)
            モノクロで学習する場合「1」、カラーで学習する場合「3」

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[学習用データ, 検証用データ, 学習用ラベル, 検証用ラベル]
        """

        class_names = Directory.get_directory_names(dataset_dir_path)

        # データセット(データとラベルのペア)を構築する
        # データ(x)には、0.0-1.0の範囲で正規化した画像データを格納する。
        # ラベル(y)には、OneHot形式[0, 1, 0]で表現したラベルを格納する。

        x = []  # データ
        y = []  # ラベル
        for idx, class_name in enumerate(class_names):
            # ラベル
            label = [0 for i in range(len(class_names))]
            label[idx] = 1
            # label = idx
            # データ
            search_pattern = "{dataset_dir_path}/{class_name}/*.png".format(
                dataset_dir_path=dataset_dir_path, class_name=class_name
            )
            file_paths = Directory.get_files(search_pattern)
            for file_path in file_paths:

                # Q. 画像はPillow形式で扱うべきか？OpenCV形式で扱うべきか？
                # A. リサイズ時の補完アルゴリズムに違いがあり、出力結果は視覚的に似ている場合も、
                #    推測結果に大きな違いを引き起こす可能性があるのでPillow推奨らしい。
                #    (つまり異なる画像処理エンジンを混ぜなければいいだけでは？)
                #
                # OpenCV(Mat)形式とPillow(PIL)形式の違い
                # ・KerasはPillow形式を基本としている。
                # ・PillowはRGB、OpenCVはBGR。
                # ・PillowとOpenCVでリサイズアルゴリズムに違いがある。
                # ・Pillowのカラー画像は(h,w,3)の形式、OpenCVは(h,w,3)の形式
                # ・Pillowのモノクロ画像は(h,w,1)の形式、OpenCVは(h,w)の形式

                # Pillow(PIL)で画像を読み込む
                color_mode = "grayscale" if input_image_channel == 1 else "rgb"
                image = load_img(file_path, color_mode=color_mode, target_size=(input_image_width, input_image_height))
                image = img_to_array(image)
                image = image.astype('float32') / 255.0
                x.append(image)
                y.append(label)

                # OpenCV(MAT)で画像を読み込む
                # as_gray = True if input_image_channel == 1 else False
                # image = ImageEditor.load_as_mat(file_path, as_gray)
                # if input_image_channel == 3:
                #     image = ImageEditor.bgr_to_rgb(image)
                # image = ImageEditor.resize(image, input_image_width, input_image_height)
                # image = ImageEditor.mat_to_3d_array(image)
                # image = ImageEditor.normalize(image)
                # x.append(image)
                # y.append(label)

        x = np.array(x)
        y = np.array(y)

        # x = x.astype('float32') / 255.0
        # from keras.utils import to_categorical
        # y = to_categorical(y, len(class_names))

        # データセットを学習用と検証用に分ける
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

        return (x_train, x_test, y_train, y_test)

    def save_dataset(self, dataset: Tuple[NDArray, NDArray, NDArray, NDArray], dataset_npy_path: str):
        """
        データセット(dataset.npy)を保存する

        Parameters
        ----------
        dataset : Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[学習用データ, 検証用データ, 学習用ラベル, 検証用ラベル]
        dataset_npy_path : str
            データセットのファイルパス
        """
        # データセットを保存する
        np.save(dataset_npy_path, dataset)

    def load_dataset(self, dataset_npy_path: str) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        データセット(dataset.npy)を読み込む

        Parameters
        ----------
        dataset_npy_path : str
            データセットのファイルパス

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[学習用データ, 検証用データ, 学習用ラベル, 検証用ラベル]
        """
        # データセットを読み込む
        x_train, x_test, y_train, y_test = np.load(dataset_npy_path)
        return (x_train, x_test, y_train, y_test)

    def train(self, model: Sequential, dataset: Tuple[NDArray, NDArray, NDArray, NDArray], data_argumentation: bool = False):
        """
        モデルを学習する

        Parameters
        ----------
        model : Sequential
            モデル
        dataset : Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[学習用データ, 検証用データ, 学習用ラベル, 検証用ラベル]
        data_argumentation : bool, optional
            データ拡張を行う場合はTrue、行わない場合はFalse。
        """
        # データ拡張を行う場合
        if data_argumentation:

            # データ拡張の方法が悪いのか精度が酷い

            # データセットを読み込む
            x_train, x_test, y_train, y_test = dataset

            # 学習用データのジェネレータ
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode="nearest")
            train_generator = train_datagen.flow(x_train, y_train, batch_size=20)

            # 検証用データのジェネレータ
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow(x_test, y_test, batch_size=20)

            # データ拡張で得られたデータを表示する
            row = 4
            col = 4
            plot_num = 1
            plt.figure("show picture that created by data arguementation.")
            for data_batch, label_batch in train_generator:
                plt.subplot(row, col, plot_num)
                plt.subplots_adjust(wspace=0.4, hspace=0.4)
                plt.tick_params(labelbottom="off")  # x軸の削除
                plt.tick_params(labelleft="off")    # y軸の削除
                # plt.title(self.categories[label_batch.tolist()[0].index(max(label_batch.tolist()[0]))])
                plt.imshow(array_to_img(data_batch[0]))
                if plot_num == row * col:
                    break
                plot_num += 1
            plt.show()

            start_time = time.time()

            # モデルを学習する
            history = model.fit(
                train_generator,
                steps_per_epoch=100,
                epochs=30,
                validation_data=test_generator,
                validation_steps=50,
                max_queue_size=300)

            # モデルの学習結果を評価する
            score = model.evaluate(x_test, y_test)

            # モデルの学習結果を表示する
            self._logger.info("Loss: {} (損失関数値 - 0に近いほど正解に近い)".format(score[0]))
            self._logger.info("Accuracy: {}% (精度 - 100% に近いほど正解に近い)".format(score[1] * 100))
            self._logger.info("Computation time: {0:.3f} sec (計算時間)".format(time.time() - start_time))
            if self._verbose:
                th_view = TrainHistoryView()
                th_view.show(history)

        # データ拡張を行わない場合
        else:

            # データセットを読み込む
            x_train, x_test, y_train, y_test = dataset

            start_time = time.time()

            # モデルを学習する
            # history = model.fit(x_train, y_train, batch_size=100, epochs=20, verbose=1, validation_data=(x_test, y_test))
            history = model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))

            # モデルの学習結果を評価する
            score = model.evaluate(x_test, y_test)

            # モデルの学習結果を表示する
            self._logger.info("Loss: {} (損失関数値 - 0に近いほど正解に近い)".format(score[0]))
            self._logger.info("Accuracy: {}% (精度 - 100% に近いほど正解に近い)".format(score[1] * 100))
            self._logger.info("Computation time: {0:.3f} sec (計算時間)".format(time.time() - start_time))
            if self._verbose:
                th_view = TrainHistoryView()
                th_view.show(history)

    def predict(self, model: Sequential, class_names: List[str], mat_bgr: Mat) -> Tuple[str, int]:
        """
        学習済みのモデルを利用し、画像のカテゴリを推測します。

        Parameters
        ----------
        model: Sequential
            学習済みのモデル
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
        input_image_width = model.layers[0].input_shape[1]
        input_image_height = model.layers[0].input_shape[2]
        input_image_channel = model.layers[0].input_shape[3]

        # 出力層の次元数(画像のカテゴリ数)
        output_dimensions = model.layers[-1].units

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
        start_time = time.time()
        predictions = model.predict(np.array([eval_image]))
        stop_time = time.time()
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
