import pygame
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.file import File
from speedfighter.utils.path import Path


class SpeedSpeaker(AppBase):
    """
    スピードスピーカー
    """

    def __init__(self):
        super().__init__()
        pygame.mixer.init()
        pygame.mixer.music.set_volume(1.0)

    @property
    def is_busy(self) -> bool:
        """
        音声を再生中かどうか
        """
        return pygame.mixer.music.get_busy()

    def play_sound(self, file_path: str):
        """
        音声を再生する

        Parameters
        ----------
        file_path : str
            音声ファイルのパス
        """
        if File.exists(file_path):
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)  # ms
                # self._logger.info("Playing...")
            # self._logger.info("Finished.")
        else:
            self._logger.error("Sound file not found. {}".format(file_path))

    def speak_number(self, number: int):
        """
        数字を読み上げる

        Parameters
        ----------
        number : int
            数字
        """
        file_path = Path.join(
            self.project_root_dir_path, "assets/voice/number/{:0=3}.mp3".format(number)
        )
        self.play_sound(file_path)

    def speak_alphabet(self, alphabet: str):
        """
        アルファベットを読み上げる

        Parameters
        ----------
        alphabet : str
            アルファベット
        """
        file_path = Path.join(
            self.project_root_dir_path, "assets/voice/alphabet/{}.mp3".format(alphabet)
        )
        self.play_sound(file_path)

    def speak_text(self, text: str):
        """
        テキストを読み上げる

        Parameters
        ----------
        text : str
            テキスト
        """
        file_path = Path.join(
            self.project_root_dir_path, "assets/voice/text/{}.mp3".format(text)
        )
        self.play_sound(file_path)
