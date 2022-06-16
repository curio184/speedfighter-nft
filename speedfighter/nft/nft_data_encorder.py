import json
from typing import Dict, List, Tuple

from speedfighter.utils.file import File


class NFTDataEncoder:

    @staticmethod
    def encode_file_base64_to_messages(file_base64: str, metadata_overview: Dict) -> List[str]:
        """
        ファイルをメッセージにエンコードする

        メッセージのフォーマット
        00000#{メタデータ}
        00001#{データ[0]}
        ...
        00100#{メタデータ}
        00101#{データ[99]}}
        ...
        """

        # メタデータ行
        metadata_record = json.dumps(metadata_overview["value"])

        # データ行
        unit = 1023 - 6
        data_records = [
            "{0:05d}#".format(idx) + file_base64[pos:pos+unit]
            for idx, pos in enumerate(range(0, len(file_base64), unit))
        ]

        # メッセージを組み立てる
        messages: List[str] = []
        for idx, data_record in enumerate(data_records):
            # 100行ごとにメタデータ行を挿入する
            if idx % 100 == 0:
                messages.append(metadata_record)
            messages.append(data_record)

        return messages

    @staticmethod
    def decode_messages_to_file_base64(messages: List[str]) -> str:
        """
        メッセージをファイルにデコードする
        """

        # メッセージをメタデータ行とデータ行に分解する
        metadata_records: List[str] = []
        data_records: List[Tuple[int, str]] = []
        for message in messages:
            if message[0:5].isnumeric() and message[5:6] == "#":
                data_records.append((int(message[0:5]), message[6:]))
            else:
                metadata_records.append(json.loads(message))

        # メタデータ行を検証する
        file_hash = ""
        for idx, metadata_record in enumerate(metadata_records):
            if idx == 0:
                file_hash = metadata_record["file_hash"]
            else:
                if metadata_record["file_hash"] != file_hash:
                    raise Exception("Detect message of different file.")

        # データ行を検証する
        file_base64 = ""
        data_records = sorted(data_records, key=lambda x: x[0])
        for idx, data_record in enumerate(data_records):
            if data_record[0] != idx:
                raise Exception("Part of the message is missing.")
            file_base64 = file_base64 + data_record[1]

        if File.get_file_base64_hash(file_base64) != file_hash:
            raise Exception("File hashes do not match.")

        return file_base64
