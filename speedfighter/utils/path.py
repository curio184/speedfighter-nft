import os


class Path:
    """
    パス
    """

    @staticmethod
    def get_file_name(file_path: str) -> str:
        file_name = os.path.basename(file_path)
        return file_name

    @staticmethod
    def get_dir_name(file_path: str) -> str:
        dir_name = os.path.dirname(file_path)
        return dir_name

    @staticmethod
    def get_extension(file_path: str) -> str:
        file_name = os.path.basename(file_path)
        if "." in file_name:
            return ".".join(file_name.split(".")[1:])
        else:
            return ""

    @staticmethod
    def get_filename_without_extension(file_path: str) -> str:
        file_name = os.path.basename(file_path)
        return file_name.split(".")[0]

    @staticmethod
    def exists(file_path: str) -> bool:
        return os.path.exists(file_path)

    def join(*paths) -> str:
        return os.path.join(*paths)
