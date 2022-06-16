
import logging
import logging.config
import os
import sys


class AppBase:

    def __init__(self):

        # ログの設定ファイルのパス
        logging_config_path = os.path.join(
            self.project_root_dir_path, "configs", "logging.ini"
        )
        # ログの出力ディレクトリ
        logs_dir_path = os.path.join(self.project_root_dir_path, "logs")
        os.makedirs(logs_dir_path, exist_ok=True)
        # ログファイル名
        log_file_name = os.path.join(
            logs_dir_path,
            os.path.splitext(os.path.basename(sys.argv[0]))[0] + ".log"
        )
        # NOTE:
        # Windows環境下でLoggerの内部処理に問題があるため\\を/に置換する。
        # 内部処理で\\が\に置換さるため\r..のようなパスを渡すとエラーになる。
        log_file_name = log_file_name.replace("\\", "/")
        # ロガーを初期化する
        logging.config.fileConfig(
            logging_config_path, defaults={"log_file_name": log_file_name}
        )
        self._logger = logging.getLogger()

    @property
    def project_root_dir_path(self) -> str:
        return os.path.dirname(os.path.dirname(__file__))
