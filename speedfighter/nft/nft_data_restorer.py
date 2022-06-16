import json
from binascii import hexlify, unhexlify
from typing import Dict, List, Tuple

from speedfighter.nft.catapult_restapi import CatapultRESTAPI
from speedfighter.nft.nft_data_encorder import NFTDataEncoder
from speedfighter.nft.nft_utils import IdConverter, KeyGenerator
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.file import File
from speedfighter.utils.inifile import INIFile
from speedfighter.utils.path import Path


class NFTDataRestorer(AppBase):

    def __init__(self) -> None:
        super().__init__()

        # INI設定
        ini_file_path = Path.join(self.project_root_dir_path, "configs/symbol.ini")
        ini_file = INIFile(ini_file_path)

        # ネットワーク情報
        self._network_name = ini_file.get_str("network", "network_name")
        self._node_url = ini_file.get_str("network", "node_url")

    def restore_nft_data(self, mosaic_id: str, file_path: str):
        """
        NFTデータをSymbolチェーンから復元する

        Parameters
        ----------
        mosaic_id : str
            モザイクID
        file_path : str
            ファイルの保存先
        """

        catapult_api = CatapultRESTAPI(self._node_url)

        # モザイクのメタデータを取得する
        mosaic_metadata = catapult_api.get_mosaic_metadata(mosaic_id)

        # モザイクのメタデータからNFT概要と参照データを取得する
        overview, tx_hashes = self._parse_mosaic_metadata(mosaic_metadata)

        messages: List[str] = []
        for tx_hash in tx_hashes:

            # 参照データのトランザクション情報を取得する
            aggregate_tx_info = catapult_api.get_confirmed_transaction_info(tx_hash)

            # トランザクション情報からメッセージを取得する
            for tx_info in aggregate_tx_info["transaction"]["transactions"]:
                message = tx_info["transaction"]["message"]
                message = unhexlify(message.encode("utf-8"))[1:].decode("utf-8")
                messages.append(message)

        # メッセージをファイルにデコードする
        file_base64 = NFTDataEncoder.decode_messages_to_file_base64(messages)

        # Base64でエンコードされたファイルを保存する
        File.save_file_base64(file_path, file_base64)

    def _parse_mosaic_metadata(self, mosaic_metadata: Dict) -> Tuple[Dict, List[str]]:
        """
        モザイクのメタデータからNFT概要と参照データを取得する

        Parameters
        ----------
        mosaic_metadata : Dict
            モザイクのメタデータ

        Returns
        -------
        Tuple[Dict, List[str]]
            (NFT概要, 参照データのトランザクションハッシュ)
        """

        key_overview = IdConverter.decimal_int_to_hex_str(KeyGenerator.generate_uint64_key("overview"))
        key_data_length = IdConverter.decimal_int_to_hex_str(KeyGenerator.generate_uint64_key("data_length"))

        # メタデータ(NFT概要)
        metadata_key_values = list(filter(
            lambda x: x["metadataEntry"]["scopedMetadataKey"] == key_overview,
            mosaic_metadata["data"]
        ))
        if len(metadata_key_values) != 1:
            raise Exception("Not Found Metadata Key")
        metadata_value = metadata_key_values[0]["metadataEntry"]["value"]
        metadata_overview = json.loads(unhexlify(metadata_value.encode("utf-8")).decode("utf-8"))

        # メタデータ(参照データ数)
        metadata_key_values = list(filter(
            lambda x: x["metadataEntry"]["scopedMetadataKey"] == key_data_length,
            mosaic_metadata["data"]
        ))
        if len(metadata_key_values) != 1:
            raise Exception("Not Found Metadata Key")
        metadata_value = metadata_key_values[0]["metadataEntry"]["value"]
        metadata_data_length = int(unhexlify(metadata_value.encode("utf-8")).decode("utf-8"))

        # メタデータ(参照データ)
        transaction_hashes: List[str] = []
        for i in range(0, metadata_data_length):
            key_data = IdConverter.decimal_int_to_hex_str(KeyGenerator.generate_uint64_key("data" + str(i)))
            metadata_key_values = list(filter(
                lambda x: x["metadataEntry"]["scopedMetadataKey"] == key_data,
                mosaic_metadata["data"]
            ))
            if len(metadata_key_values) != 1:
                raise Exception("Not Found Metadata Key")
            metadata_value = metadata_key_values[0]["metadataEntry"]["value"]
            transaction_hashes.extend(json.loads(unhexlify(metadata_value.encode("utf-8")).decode("utf-8")))

        return (metadata_overview, transaction_hashes)
