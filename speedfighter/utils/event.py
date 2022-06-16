from __future__ import annotations

from queue import Queue


class EventArgs:
    """
    イベント引数
    """

    def __init__(self, params: dict = {}):
        self._params = params

    @property
    def params(self) -> dict:
        return self._params


class EventHandler:
    """
    イベントハンドラー
    """

    def __init__(self, obj):
        max_size = 10
        self._queue = Queue(max_size)
        self._obj = obj
        self._funcs = []

    def add(self, func) -> EventHandler.listen:
        """
        callbackメソッドを登録する
        """
        self._funcs.append(func)
        return self.listen

    def remove(self, func):
        """
        callbackメソッドを登録解除する
        """
        self._funcs.remove(func)

    def fire(self, eargs: EventArgs):
        """
        eventを実行する
        """
        self._queue.put(eargs)

    def listen(self):
        """
        eventを受けとる
        """
        if not self._queue.empty():
            eargs = self._queue.get()
            for func in self._funcs:
                func(self._obj, eargs)
