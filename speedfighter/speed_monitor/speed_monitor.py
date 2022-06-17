import datetime as dt
import time
from threading import Event, Thread
from typing import List, Tuple

import cv2
from cv2 import Mat
from speedfighter.digit_recognition.image_classifier_lite import \
    ImageClassifierLite
from speedfighter.digit_recognition.seven_seg_splitter import SevenSegSplitter
from speedfighter.speed_monitor.fps_counter import FPSCounter
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.directory import Directory
from speedfighter.utils.event import EventArgs, EventHandler
from speedfighter.utils.image_editor import ImageEditor
from speedfighter.utils.inifile import INIFile
from speedfighter.utils.path import Path


class SpeedMonitor(AppBase, Thread):
    """
    スピードを監視する
    """

    def __init__(self, should_stop: Event):

        AppBase.__init__(self)
        Thread.__init__(self)
        self.name = "SpeedMonitorThread"

        # INI設定
        ini_file_path = Path.join(self.project_root_dir_path, "configs/speedfighter.ini")
        ini_file = INIFile(ini_file_path)
        self._model_tflite_path = ini_file.get_str("speedmonitor", "model_tflite_path")
        self._class_names_json_path = ini_file.get_str("speedmonitor", "class_names_json_path")
        self._preview = ini_file.get_bool("speedmonitor", "preview")
        self._record = ini_file.get_bool("speedmonitor", "record")
        self._capture_devide = ini_file.get_str("speedmonitor", "capture_device")
        self._capture_source_dir = ini_file.get_str("speedmonitor", "capture_source_dir")

        # スレッド制御
        self._should_stop = should_stop

        # イベント制御
        self._speed_detected_eventhandler = EventHandler(self, 10)

        # VideoCaptureオブジェクトを初期化
        if self._capture_devide == "DummyVideoCapture":
            self._capture = DummyVideoCaputure(self._capture_source_dir)
        else:
            self._capture = cv2.VideoCapture(0)
            if not self._capture.isOpened():
                raise Exception("Failed to initialize video capture device.")

    def _release(self):
        cv2.destroyAllWindows()
        self._capture.release()

    def run(self):

        self._logger.info("Speed monitor started.")

        # 画像保存ディレクトリ
        save_path = Path.join(
            self.project_root_dir_path,
            "assets/rawdata/{:%y%m%d_%H%M%S}/".format(dt.datetime.now())
        )
        if self._record and not Path.exists(save_path):
            try:
                Directory.create_directory(save_path)
            except Exception as ex:
                self._logger.error(ex)

        # 画像分類器
        classifier = ImageClassifierLite()
        interpreter = classifier.setup_interpreter(self._model_tflite_path)
        class_names = classifier.load_class_names(self._class_names_json_path)

        # FPSカウンター
        fps_counter = FPSCounter()

        while True:

            # 停止命令
            if self._should_stop.is_set():
                break

            # フレームを読み込む
            ret, mat_gbr = self._capture.read()
            if not ret:
                continue

            # 画像を区域ごとに分割する
            splitter = SevenSegSplitter()
            mat_digits = splitter.split_into_digits(mat_gbr)

            # 画像のカテゴリを推測する
            pre_numbers: List[str] = []
            pre_accuracies: List[int] = []
            for mat_digit in mat_digits:
                pre_number, pre_accuracy = classifier.predict(interpreter, class_names, mat_digit)
                pre_numbers.append(pre_number)
                pre_accuracies.append(pre_accuracy)

            # 推測結果を描画する
            mat_gbr = splitter.draw_bounding_box(mat_gbr, pre_numbers)

            # フレームレートを計算し描画する
            fps = fps_counter.count()
            self._logger.info("{:.1f}fps".format(fps))
            ImageEditor.draw_text(mat_gbr, "{:.1f}fps".format(fps), (50, 50))

            # 推測結果をスピードに変換する
            speed = self._prediction_numbers_to_speed(pre_numbers, pre_accuracies)

            # スピードを検出できた場合
            if speed > 0:
                # スピード検出イベントを実行する
                self._on_speed_detected(EventArgs({"speed": speed, "capture": mat_gbr}))

            # プレビュー表示する
            if self._preview:
                # フレームを表示する
                cv2.imshow("Capture", mat_gbr)

                # 33msec待機する(30fps)
                key_press = cv2.waitKey(33)

                # ESC押下でプログラムを終了する
                if key_press == 27:
                    break

            # 画像を保存する
            if self._record:
                file_path = "{}{:%y%m%d_%H%M%S%f}.jpg".format(save_path, dt.datetime.now())
                cv2.imwrite(file_path, mat_gbr)

            # time.sleep(0.1)

        self._logger.info("Speed monitor stopped.")
        self._release()

    def _prediction_numbers_to_speed(self, prediction_numbers: List[str], prediction_accuracies: List[int]) -> int:
        """
        推測結果をスピードに変換する

        Parameters
        ----------
        prediction_numbers : List[str]
            推測結果のリスト
        prediction_accuracies : List[int]
            推測精度のリスト

        Returns
        -------
        int
            スピード
        """

        if "no_signal" in prediction_numbers:
            return 0

        # 推測精度が閾値を満たさない場合、無視する
        for pre_accuracy in prediction_accuracies:
            if pre_accuracy < 80:
                return 0

        pattern = [str.isdigit(x) for x in prediction_numbers]
        if pattern == [True, True, True]:
            return int(prediction_numbers[0] + prediction_numbers[1] + prediction_numbers[2])
        elif pattern == [False, True, True]:
            return int(prediction_numbers[1] + prediction_numbers[2])
        elif pattern == [False, False, True]:
            return int(prediction_numbers[2])
        else:
            return 0

    def _on_speed_detected(self, eargs: EventArgs):
        """
        スピードを検出したとき
        """
        self._speed_detected_eventhandler.fire(eargs)


class DummyVideoCaputure:
    """
    VideoCaptureのダミークラス(リプレイテストで使用)
    """

    def __init__(self, dir_path: str) -> None:
        search_pattern = "{}/**.jpg".format(dir_path)
        self._file_paths = sorted(Directory.get_files(search_pattern))
        self._file_index = 0

    def isOpened(self) -> bool:
        if len(self._file_paths) >= 1:
            return True
        else:
            return False

    def release(self):
        pass

    def read(self) -> Tuple[bool, Mat]:
        file_path = self._file_paths[self._file_index]
        mat_gbr = ImageEditor.load_as_mat(file_path)
        if self._file_index == len(self._file_paths) - 1:
            self._file_index = 0
        else:
            self._file_index = self._file_index + 1
        return (True, mat_gbr)


if __name__ == "__main__":

    should_stop = Event()

    # スピードガンの速度を監視する
    monitor = SpeedMonitor(should_stop)
    monitor.start()

    time.sleep(180)
    should_stop.set()
