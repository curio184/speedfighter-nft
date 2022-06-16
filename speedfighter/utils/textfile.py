import codecs


class TextFile:

    @staticmethod
    def load(file_path: str) -> str:
        """
        テキストとして保存する
        """
        with codecs.open(file_path, "r", "utf8") as f:
            text = f.read()
        return text

    @staticmethod
    def save(file_path: str, text: str):
        """
        テキストとして保存する
        """
        with codecs.open(file_path, "w", "utf8") as f:
            f.write(text)
