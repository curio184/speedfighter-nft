import datetime as dt

import cv2
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.directory import Directory
from speedfighter.utils.image_editor import ImageEditor
from speedfighter.utils.path import Path


class VideoCamera(AppBase):

    def __init__(self):
        super().__init__()

        # VideoCaptureオブジェクトを初期化
        self._capture = cv2.VideoCapture(0)
        if self._capture.isOpened() is False:
            raise Exception("Failed to initialize video capture device.")

    def __del__(self):
        self._capture.release()

    def record(self, save_path: str, preview_only: bool = False):
        """
        録画を開始する
        """

        # 画像保存ディレクトリを作成する
        Directory.create_directory(save_path)

        # 画面を表示する
        cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

        while True:

            # フレームを読み込む
            ret, frame = self._capture.read()
            if not ret:
                continue

            # フレームを表示する
            cv2.imshow("Capture", frame)

            # 画像を保存する
            if not preview_only:
                file_path = "{}{:%y%m%d_%H%M%S%f}.jpg".format(save_path, dt.datetime.now())
                cv2.imwrite(file_path, frame)

            # 33msec待機する(30fps)
            key_press = cv2.waitKey(33)

            # ESC押下でプログラムを終了する
            if key_press == 27:
                break

        # 画面を非表示にする
        cv2.destroyAllWindows()

    def play(self, dir_path: str):
        """
        録画を再生する

        Parameters
        ----------
        dir_path : str
            録画のディレクトリパス
        """

        # 画像ファイルを読み込む
        search_pattern = "{}/**.jpg".format(dir_path)
        file_paths = sorted(Directory.get_files(search_pattern))

        # 順に再生する
        for idx, file_path in enumerate(file_paths):

            frame = ImageEditor.load_as_mat(file_path)

            # フレームを表示する
            cv2.imshow("Capture", frame)

            # 33msec待機する(30fps)
            key_press = cv2.waitKey(33)

            # ESC押下でプログラムを終了する
            if key_press == 27:
                break
