import base64
import hashlib
import os
import shutil

import cv2
import numpy as np
from cv2 import Mat
from numpy.typing import NDArray


class File:

    @staticmethod
    def move(source: str, dest: str):
        try:
            shutil.move(source, dest)
        except:
            print("ファイルの移動に失敗しました。{0}->{1}".format(source, dest))

    @staticmethod
    def exists(file_path: str) -> bool:
        return os.path.exists(file_path)

    @staticmethod
    def load_as_base64(file_path: str) -> str:
        """
        ファイルをBase64でエンコードし文字列として読み込む
        """
        with open(file_path, "rb") as f:
            file_binary = base64.b64encode(f.read())
        return file_binary.decode('utf-8')

    @staticmethod
    def save_file_base64(file_path: str, file_base64: str):
        """
        Base64でエンコードされたファイルを保存する
        """
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(file_base64))

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """
        ファイルのハッシュ(sha256)を取得する
        """
        with open(file_path, "rb") as f:
            hash_sha256 = hashlib.sha256(f.read()).hexdigest()
        return hash_sha256

    @staticmethod
    def get_file_base64_hash(file_base64: str) -> str:
        """
        Base64でエンコードされたファイルのハッシュ(sha256)を取得する
        """
        file_binary = base64.b64decode(file_base64)
        return hashlib.sha256(file_binary).hexdigest()

    @staticmethod
    def base64_to_mat(file_base64: str) -> Mat:
        """
        Base64でエンコードされたファイルをMatに変換する
        """
        file_binary = base64.b64decode(file_base64)
        file_array: NDArray = np.frombuffer(file_binary, dtype=np.uint8)
        mat = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        return mat
