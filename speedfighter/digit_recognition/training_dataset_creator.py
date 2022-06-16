import codecs
import os
from typing import List

from cv2 import Mat
from speedfighter.digit_recognition.seven_seg_splitter import SevenSegSplitter
from speedfighter.digit_recognition.views.class_selection_view import \
    ClassSelectionView
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.directory import Directory
from speedfighter.utils.image_editor import ImageEditor
from speedfighter.utils.path import Path


class TrainingDatasetCreator(AppBase):
    """
    素材画像をカテゴリに分類し、データセットを作成する
    """

    def __init__(self):
        super().__init__()

    def create_dataset(self, class_names: List[str], rawdata_dir_path: str):
        """
        素材画像をカテゴリに分類し、データセットを作成する

        Parameters
        ----------
        class_names : List[str]
            カテゴリ一覧
        rawdata_dir_path : str
            素材画像のファイルパス一覧
        """

        # カテゴリごとにディレクトリを作成する
        self._create_class_name_directories(class_names)

        # 素材画像を取得する
        search_pattern = rawdata_dir_path + "*.jpg"
        file_paths = Directory.get_files(search_pattern)

        # 分類済みを除外する(レジューム機能)
        complete_files = self._get_complete_files(rawdata_dir_path)
        file_paths = list(filter(lambda x: Path.get_file_name(x) not in complete_files, file_paths))

        selections = []
        for file_path in file_paths:

            # 素材画像を読み込む
            mat = ImageEditor.load_as_mat(file_path)
            file_name = Path.get_file_name(file_path)

            # 7-Seg画像を数字ごとに分割する
            splitter = SevenSegSplitter()
            mat_bounding = splitter.draw_bounding_box(mat)
            mats_digits = splitter.split_into_digits(mat)

            # 素材画像をカテゴリに分類する
            selections = list(map(lambda x: "b" if x == "blank" else x, selections))
            selections = list(map(lambda x: "n" if x == "no_signal" else x, selections))
            cs_view = ClassSelectionView(list(selections))
            selections = cs_view.show(mat_bounding, file_name, class_names)
            selections = list(map(lambda x: "blank" if x == "b" else x, selections))
            selections = list(map(lambda x: "no_signal" if x == "n" else x, selections))

            # 素材画像をカテゴリごとに保存する
            self._save_image_by_class_name(mats_digits, selections, file_path)

            # 分類済みを追加する(レジューム機能)
            self._add_complete_file(rawdata_dir_path, file_path)

    def _create_class_name_directories(self, class_names: List[str]):
        """
        カテゴリごとにディレクトリを作成する

        Parameters
        ----------
        class_names : List[str]
             カテゴリ一覧
        """

        # カテゴリごとにディレクトリを作成する
        for class_name in class_names:
            path = Path.join(
                self.project_root_dir_path, "assets/dataset/" + class_name
            )
            try:
                Directory.create_directory(path)
            except Exception as ex:
                self._logger.error(ex)

    def _save_image_by_class_name(self,  mats: List[Mat], class_names: List[str], file_path: str):
        """
        素材画像をカテゴリごとに保存する

        Parameters
        ----------
        mats : List[Mat]
            素材画像
        class_names : List[str]
            カテゴリ
        file_path : str
            ファイル名
        """

        for idx, mat in enumerate(mats):
            try:
                classified_file_path = Path.join(
                    self.project_root_dir_path,
                    "assets/dataset/{}/{}_{}.png".format(
                        class_names[idx],
                        Path.get_filename_without_extension(file_path),
                        idx
                    )
                )
                ImageEditor.save(mat, classified_file_path)
            except Exception as ex:
                self._logger.error(ex)

    def _get_complete_files(self, rawdata_dir_path: str) -> List[str]:
        complete_files = []
        try:
            with codecs.open(rawdata_dir_path + "complete.txt", "r", "utf8") as f:
                complete_files = f.read().splitlines()
        except Exception as ex:
            self._logger.error(ex)
        return complete_files

    def _add_complete_file(self, rawdata_dir_path: str, file_path: str):
        try:
            with codecs.open(rawdata_dir_path + "complete.txt", "a", "utf8") as f:
                f.write(Path.get_file_name(file_path) + "\n")
        except Exception as ex:
            self._logger.error(ex)
