import os
from glob import glob
from typing import List


class Directory:
    """
    ディレクトリ
    """

    @staticmethod
    def get_files(search_pattern: str) -> List[str]:
        return glob(search_pattern)

    @staticmethod
    def get_directories(dir_path: str) -> List[str]:
        return list(map(lambda x: os.path.join(dir_path, x), os.listdir(dir_path)))

    @staticmethod
    def get_directory_names(dir_path: str) -> List[str]:
        return os.listdir(dir_path)

    @staticmethod
    def create_directory(path: str):
        try:
            os.makedirs(path, exist_ok=True)
        except:
            raise Exception("ディレクトリ'{0}'の作成に失敗しました。".format(path))
