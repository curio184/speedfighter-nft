import codecs
import json


class JsonFile:

    @staticmethod
    def save(file_path: str, data: dict):
        """
        JSONとして保存する
        """
        with codecs.open(file_path, "w", "utf8") as f:
            json.dump(data, f, ensure_ascii=False)

    @staticmethod
    def load(file_path: str) -> dict:
        """
        JSONを読み込む
        """
        data = None
        with codecs.open(file_path, "r", "utf8") as f:
            data = json.load(f)
        return data
