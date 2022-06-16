import time
from datetime import datetime

import sha3


class NonceGenerator:

    @staticmethod
    def generate() -> int:
        """
        ナンスを生成する
        """
        return int(time.mktime(datetime.now().timetuple()))


class KeyGenerator:

    @staticmethod
    def generate_uint64_key(input: str) -> int:
        # 下記コマンドと互換 (と思いきや先頭桁のみ稀に違う、、、)
        # $ symbol-cli converter stringToKey -v header
        # AD6D8491D21180E5D
        hasher = sha3.sha3_256()
        hasher.update(input.encode("utf-8"))
        digest = hasher.digest()
        result = int.from_bytes(digest[0:8], "little")
        return result


class IdConverter:

    @staticmethod
    def decimal_int_to_hex_str(decimal_int: int) -> str:
        return "{:0>16}".format(hex(decimal_int)[2:].upper())

    @staticmethod
    def hex_str_to_decimal_int(hex_str: str) -> int:
        return int(hex_str, 16)
