import datetime as dt
from typing import List, Tuple, Union

from cv2 import Mat
from speedfighter.utils.app_base import AppBase


class SpeedRecorder(AppBase):
    """
    スピードレコーダー
    """

    def __init__(self):
        super().__init__()
        self._max_records = 150
        self._records: List[Tuple[dt.datetime, int, Mat]] = []
        self._fastest_record: Tuple[dt.datetime, int, Mat] = None

    def add_record(self, speed: int, mat: Mat):
        """
        レコードを追加する

        Parameters
        ----------
        speed : int
            速度
        mat : Mat
            画像
        """
        this_record = (dt.datetime.now(), speed, mat)

        self._records.append(this_record)

        # 古いレコードを削除する
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        # ベストレコードを更新する
        if (self._fastest_record == None) or (speed > self._fastest_record[1]):
            self._fastest_record = this_record

    def get_latest_fastest_record(self, sample_period: int = 1, min_sample_size: int = 5) -> Union[Tuple[dt.datetime, int, Mat], None]:
        """
        直近の最速レコードを取得する

        Parameters
        ----------
        sample_period : int, by default 1
            サンプル期間(秒)
        min_sample_size : int, optional, by default 5
            最小サンプル数

        Returns
        -------
        Union[Tuple[dt.datetime, int, Mat], None]
            直近の最速レコード
        """
        now = dt.datetime.now()
        targets = list(filter(lambda x: (now - x[0]).total_seconds() <= sample_period, self._records))
        if len(targets) < min_sample_size:
            return None
        else:
            return max(targets, key=lambda x: x[1])

    def get_fastest_record(self) -> Union[Tuple[dt.datetime, int, Mat], None]:
        """
        最速レコードを取得する
        """
        return self._fastest_record

    def reset_records(self):
        """
        レコードをリセットする
        """
        self._records = []
        self._fastest_record = None
