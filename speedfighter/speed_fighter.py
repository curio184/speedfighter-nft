import datetime as dt
import signal
import sys
from abc import ABC, abstractmethod
from threading import Event
from time import sleep

import requests
import RPi.GPIO as GPIO

from speedfighter.nft.nft_creator import NFTCreator
from speedfighter.speed_monitor.speed_monitor import SpeedMonitor
from speedfighter.speed_monitor.speed_recorder import SpeedRecorder
from speedfighter.speed_monitor.speed_speaker import SpeedSpeaker
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.directory import Directory
from speedfighter.utils.event import EventArgs, EventHandler
from speedfighter.utils.image_editor import ImageEditor
from speedfighter.utils.inifile import INIFile
from speedfighter.utils.path import Path
from speedfighter.utils.threading import ThreadWithReturnValue
from speedfighter.utils.twitter import Twitter

PIN_LED_1 = 4       # 電源LED
PIN_LED_2 = 17      # 処理中LED
PIN_SWITCH_1 = 27   # 電源スイッチ
PIN_SWITCH_2 = 23   # オプションボタン1(IssueNFT, OK)
PIN_SWITCH_3 = 24   # オプションボタン2(RESET, Cancel)
PIN_RELAY = 25      # リレースイッチ


class SpeedMonitorController(AppBase):

    def __init__(self):
        super().__init__()

        # 終了要求時に実行するコールバック関数を登録する
        signal.signal(signal.SIGTERM, self._on_termination_requested)

        # GPIOの初期化
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PIN_LED_1, GPIO.OUT)
        GPIO.setup(PIN_LED_2, GPIO.OUT)
        GPIO.setup(PIN_SWITCH_1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(PIN_SWITCH_2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(PIN_SWITCH_3, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(PIN_RELAY, GPIO.OUT)
        # callbackにMainThreadのMethodを直接指定するとThreadUnsafeな仕様になっており、
        # また、callbackされたMethod内で重い処理を行うとチャタリングを引き起こすため、
        # 独自のThreadSafeなEventHandlerを通じて呼び出すようにしている。
        button_clicked_event_handler = EventHandler(self)
        GPIO.add_event_detect(PIN_SWITCH_1, GPIO.RISING, callback=button_clicked_event_handler.fire, bouncetime=200)
        GPIO.add_event_detect(PIN_SWITCH_2, GPIO.RISING, callback=button_clicked_event_handler.fire, bouncetime=200)
        GPIO.add_event_detect(PIN_SWITCH_3, GPIO.RISING, callback=button_clicked_event_handler.fire, bouncetime=200)
        self._button_clicked_event_listener = button_clicked_event_handler.add(self._on_button_clicked)
        GPIO.output(PIN_LED_1, GPIO.LOW)
        GPIO.output(PIN_LED_2, GPIO.LOW)
        GPIO.output(PIN_RELAY, GPIO.LOW)

        # アクティブダイアログ
        self._active_dialog: Dialog = None

        # スピードモニター
        self._speed_monitor: SpeedMonitor = None
        self._speed_monitor_should_stop: Event = None
        self._speed_detected_event_listener = None

        # スピードレコーダー
        self._recorder = SpeedRecorder()

        # スピードスピーカー
        self._speaker = SpeedSpeaker()

    def _release(self):
        GPIO.cleanup()
        sys.exit()

    def start(self):
        try:
            while True:
                if self._button_clicked_event_listener:
                    self._button_clicked_event_listener()
                if self._speed_detected_event_listener:
                    self._speed_detected_event_listener()
                sleep(0.01)
        except KeyboardInterrupt:
            pass
        self._release()

    def _on_button_clicked(self, sender: object, channel):

        # 電源ボタンを押したとき
        if channel == PIN_SWITCH_1:

            self._logger.info("PIN_SWITCH_1 clicked.")

            # スピードモニターが起動していないとき
            if GPIO.input(PIN_LED_1) == GPIO.LOW:

                # スピードモニターを起動する
                GPIO.output(PIN_LED_1, GPIO.HIGH)
                GPIO.output(PIN_RELAY, GPIO.HIGH)
                self._start_speed_monitor()

            # スピードモニターが起動しているとき
            else:
                # ダイアログが開かれていないとき
                if not self._active_dialog:

                    # スピードモニターを停止する
                    self._stop_speed_monitor()
                    GPIO.output(PIN_LED_1, GPIO.LOW)
                    GPIO.output(PIN_RELAY, GPIO.LOW)

                # ダイアログが開かれているとき
                else:
                    self._logger.info("Please close the dialog first.")

        # オプションボタン1をクリックしたとき
        elif channel == PIN_SWITCH_2:

            self._logger.info("PIN_SWITCH_2 clicked.")

            # ダイアログを開いていないとき
            if not self._active_dialog:

                # スピードモニターを起動しているとき
                if self._speed_monitor and self._speed_monitor.is_alive():

                    # IssueNFTRecordダイアログを開く
                    GPIO.output(PIN_LED_2, GPIO.HIGH)
                    self._active_dialog = IssueNFTDialog(self._speaker, self._recorder)
                    if not self._active_dialog.dialog_opened():
                        GPIO.output(PIN_LED_2, GPIO.LOW)
                        self._active_dialog = None

                # スピードモニターを起動していないとき
                else:
                    self._logger.info("Please start the speed monitor first.")

            # ダイアログを開いているとき
            else:

                # ダイアログのOKイベントを実行する
                self._active_dialog.ok_button_clicked()
                self._active_dialog = None
                GPIO.output(PIN_LED_2, GPIO.LOW)

        # オプションボタン2をクリックしたとき
        elif channel == PIN_SWITCH_3:

            self._logger.info("PIN_SWITCH_3 clicked.")

            # ダイアログを開いていないとき
            if not self._active_dialog:
                if self._speed_monitor and self._speed_monitor.is_alive():

                    # ResetRecordダイアログを開く
                    GPIO.output(PIN_LED_2, GPIO.HIGH)
                    self._active_dialog = ResetRecordsDialog(self._speaker, self._recorder)
                    self._active_dialog.dialog_opened()
                else:
                    self._logger.info("Please start the speed monitor first.")

            # ダイアログを開いているとき
            else:

                # ダイアログのCancelイベントを実行する
                self._active_dialog.cancel_button_clicked()
                self._active_dialog = None
                GPIO.output(PIN_LED_2, GPIO.LOW)

        else:

            self._logger.info("Undefined channel clicked. channel: {}".format(channel))

    def _start_speed_monitor(self):
        """
        スピードモニターを起動する
        """
        self._logger.info("Request to start the speed monitor.")
        self._speaker.speak_text("playball")
        self._speed_monitor_should_stop = Event()
        self._speed_monitor = SpeedMonitor(self._speed_monitor_should_stop)
        self._speed_detected_event_listener = \
            self._speed_monitor._speed_detected_eventhandler.add(self._on_speed_detected)
        self._speed_monitor.start()

    def _stop_speed_monitor(self):
        """
        スピードモニターを停止する
        """
        self._logger.info("Request to stop the speed monitor.")
        self._speed_monitor_should_stop.set()
        self._speaker.speak_text("gameset")
        self._speed_monitor.join(timeout=10)

    def _on_speed_detected(self, sender: object, eargs: EventArgs):
        """
        スピードモニターがスピードを検出したとき
        """
        self._recorder.add_record(eargs.params["speed"], eargs.params["capture"])
        latest_record = self._recorder.get_latest_fastest_record(1, 5)
        # 記録がある and 使用中でない and ダイアログが開いていない
        if latest_record and not self._speaker.is_busy and not self._active_dialog:
            self._speaker.speak_number(latest_record[1])

    def _on_termination_requested(self, signal, frame):
        """
        終了要求時に実行されるコールバック関数
        """
        self._logger.info("Stop speed fighter.")
        if self._speed_monitor_should_stop:
            self._speed_monitor_should_stop.set()
        self._release()


class Dialog(ABC):
    """
    ダイアログ
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def dialog_opened(self) -> bool:
        """
        ダイアログが開かれたとき
        """
        pass

    @abstractmethod
    def ok_button_clicked(self):
        """
        OKボタンがクリックされたとき
        """
        pass

    @abstractmethod
    def cancel_button_clicked(self):
        """
        Cancelボタンがクリックされたとき
        """
        pass


class IssueNFTDialog(Dialog, AppBase):
    """
    NFTを発行するダイアログ
    """

    def __init__(self, speed_speaker: SpeedSpeaker, speed_recorder: SpeedRecorder) -> None:
        Dialog.__init__(self)
        AppBase.__init__(self)
        self._speaker = speed_speaker
        self._recorder = speed_recorder

    def dialog_opened(self) -> bool:

        # 最速レコードを取得する
        self._fastest_record = self._recorder.get_fastest_record()

        if self._fastest_record:
            # 最高速度は{}です。保存しますか？
            self._speaker.speak_text("save_record_ask_1")
            self._speaker.speak_number(self._fastest_record[1])
            self._speaker.speak_text("save_record_ask_2")
            self._speaker.speak_text("save_record_ask_3")
            return True
        else:
            # 記録がありません
            self._speaker.speak_text("no_record")
            return False

    def ok_button_clicked(self):

        # NFTを発行しています。しばらくお待ちください。
        self._speaker.speak_text("issue_nft_wait_a_moment")

        # INI設定
        ini_file_path = Path.join(self.project_root_dir_path, "configs/symbol.ini")
        ini_file = INIFile(ini_file_path)

        try:

            # ネットワーク接続に問題がないかチェック
            network_name = ini_file.get_str("network", "network_name")
            node_url = ini_file.get_str("network", "node_url")
            max_retry_count = 3
            retry_count = 0
            retry_span = 10
            while retry_count < max_retry_count:
                try:
                    response = requests.get(node_url + "/chain/info", timeout=1)
                    if response.status_code != 200:
                        raise Exception("status code is {}".format(response.status_code))
                    break
                except Exception as ex:
                    self._logger.error(ex)
                    retry_count += 1
                    sleep(retry_span)
                    raise Exception("Unable to connect to the Internet.")

            # 画像保存ディレクトリを作成する
            request_dir_path = "./assets/nft/request/{:%y%m%d_%H%M%S}/".format(dt.datetime.now())
            Directory.create_directory(request_dir_path)

            # 画像を保存する
            nft_file_path = Path.join(request_dir_path, "nft_data.jpg")
            ImageEditor.save(self._fastest_record[2], nft_file_path)

            # NFTに記載する情報
            file_path = nft_file_path
            record_holder_address = ini_file.get_str("user_account", "address")
            record_holder_name = ini_file.get_str("user_account", "name")
            record_datetime = self._fastest_record[0]
            record_value = self._fastest_record[1]
            record_method = ini_file.get_str("user_account", "record_method")

            # NFTを発行する
            self._logger.info("Start to Issue NFT.")
            creator = NFTCreator()
            th_creator = ThreadWithReturnValue(
                target=creator.create_nft,
                args=(
                    file_path,
                    record_holder_address,
                    record_holder_name,
                    record_datetime,
                    record_value,
                    record_method
                )
            )
            th_creator.start()
            while th_creator.is_alive():
                GPIO.output(PIN_LED_2, GPIO.LOW)
                sleep(1.2)
                GPIO.output(PIN_LED_2, GPIO.HIGH)
                sleep(0.05)
                GPIO.output(PIN_LED_2, GPIO.LOW)
                sleep(1.2)
                GPIO.output(PIN_LED_2, GPIO.HIGH)
                sleep(0.05)
                GPIO.output(PIN_LED_2, GPIO.LOW)
                sleep(0.05)
                GPIO.output(PIN_LED_2, GPIO.HIGH)
                sleep(0.05)
            mosaic_id = th_creator.join()
            self._logger.info("NFT has been issued. MosaicID={}".format(mosaic_id))
            self._logger.info("https://testnet.symbol.fyi/mosaics/{}".format(mosaic_id))

            # INI設定
            ini_file_path = Path.join(self.project_root_dir_path, "configs/twitter.ini")
            ini_file = INIFile(ini_file_path)

            if ini_file.get_bool("twitter", "enabled"):
                twitter = Twitter(
                    ini_file.get_str("twitter", "consumer_key"),
                    ini_file.get_str("twitter", "consumer_secret"),
                    ini_file.get_str("twitter", "access_token"),
                    ini_file.get_str("twitter", "access_token_secret")
                )

                explorer_url = ""
                if network_name == "mainnet":
                    explorer_url = "https://symbol.fyi/mosaics/"
                else:
                    explorer_url = "https://testnet.symbol.fyi/mosaics/"
                twitter.send_picture(
                    "只今の投球は時速{}kmでした。記録は #Symbol にフルオンチェーンで保存されました。".format(record_value) +
                    "世界初NFT対応スピードガンSpeed Fighter!! {}{}".format(explorer_url, mosaic_id),
                    nft_file_path
                )

            # NFTを発行しました。
            self._speaker.speak_text("save_record_ok")
            self._speaker.speak_text("mosaic_id_is_1")
            for char in mosaic_id:
                if char.isnumeric():
                    self._speaker.speak_number(int(char))
                else:
                    self._speaker.speak_alphabet(char)
            self._speaker.speak_text("mosaic_id_is_2")

        except Exception as ex:
            self._logger.error(ex)
            self._speaker.speak_text("error")

    def cancel_button_clicked(self):
        # 保存をキャンセルしました
        self._speaker.speak_text("save_record_cancel")


class ResetRecordsDialog(Dialog):
    """
    スピードの履歴をリセットするダイアログ
    """

    def __init__(self, speed_speaker: SpeedSpeaker, speed_recorder: SpeedRecorder) -> None:
        super().__init__()
        self._speaker = speed_speaker
        self._recorder = speed_recorder

    def dialog_opened(self):
        # 記録をリセットしますか？
        self._speaker.speak_text("reset_records_ask")

    def ok_button_clicked(self):
        # レコードをリセットする
        self._recorder.reset_records()
        # 記録をリセットしました
        self._speaker.speak_text("reset_records_ok")

    def cancel_button_clicked(self):
        # リセットをキャンセルしました
        self._speaker.speak_text("reset_records_cancel")


if __name__ == "__main__":
    launcher = SpeedMonitorController()
    launcher.start()
