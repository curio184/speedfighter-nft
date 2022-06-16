
import time


class FPSCounter:
    """
    FPSカウンター
    """

    def __init__(self) -> None:
        self._histories: float = []

    def count(self) -> float:

        now = time.time()

        self._histories.append(now)

        # 古い履歴を削除する
        if len(self._histories) > 300:
            self._histories = self._histories[-300:]

        # フレームレートを計算する
        if now - self._histories[-1] > 1:
            return 1.0 / (now - self._histories[-1])
        else:
            return len(list(filter(lambda x: now - x <= 1.0, self._histories)))
