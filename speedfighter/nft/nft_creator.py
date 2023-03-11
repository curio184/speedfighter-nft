import datetime as dt
import json
from datetime import datetime, timedelta, timezone
from email.headerregistry import Address
from time import sleep
from typing import Dict, List, Tuple

import requests
from speedfighter.nft.catapult_restapi import CatapultRESTAPI
from speedfighter.nft.nft_data_encorder import NFTDataEncoder
from speedfighter.nft.nft_utils import (IdConverter, KeyGenerator,
                                        NonceGenerator)
from speedfighter.utils.app_base import AppBase
from speedfighter.utils.directory import Directory
from speedfighter.utils.file import File
from speedfighter.utils.inifile import INIFile
from speedfighter.utils.jsonfile import JsonFile
from speedfighter.utils.path import Path
from symbolchain.CryptoTypes import PrivateKey
from symbolchain.facade.SymbolFacade import SymbolFacade
from symbolchain.sc import (AggregateCompleteTransaction, Amount,
                            BlockDuration, EmbeddedMosaicDefinitionTransaction,
                            EmbeddedMosaicMetadataTransaction,
                            EmbeddedMosaicSupplyChangeTransaction,
                            EmbeddedTransferTransaction, MosaicFlags,
                            MosaicNonce, MosaicSupplyChangeAction,
                            TransferTransaction)
from symbolchain.symbol import IdGenerator
from symbolchain.symbol.KeyPair import KeyPair


class NFTCreator(AppBase):

    def __init__(self) -> None:
        super().__init__()

        # INI設定
        ini_file_path = Path.join(self.project_root_dir_path, "configs/symbol.ini")
        ini_file = INIFile(ini_file_path)

        # ネットワーク情報
        self._network_name = ini_file.get_str("network", "network_name")
        self._node_url = ini_file.get_str("network", "node_url")

        catapult_api = CatapultRESTAPI(self._node_url)
        self._epoch_adjustment = catapult_api.get_epoch_adjustment()      # 1637848847
        self._currency_mosaic_id = catapult_api.get_currency_mosaic_id()  # symbol.xym, 0x3A8416DB2D53B6C8
        self._fee_multipliers = catapult_api.get_fee_info()
        self._logger.info(f"fee_multipliers: {self._fee_multipliers}")

        # トランザクション設定
        fee_status = ini_file.get_str("transaction", "fee_status")
        fee_ratio = ini_file.get_int("transaction", "fee_ratio")
        fee_status = "averageFeeMultiplier" if "average" in fee_status.lower() else fee_status
        fee_status = "medianFeeMultiplier" if "median" in fee_status.lower() else fee_status
        fee_status = "highestFeeMultiplier" if "highest" in fee_status.lower() else fee_status
        fee_status = "lowestFeeMultiplier" if "lowest" in fee_status.lower() else fee_status
        fee_status = "minFeeMultiplier" if "min" in fee_status.lower() else fee_status
        self._max_fee_per_tx = self._fee_multipliers[fee_status] * fee_ratio
        self._max_fee_per_aggregate_tx = ini_file.get_int("transaction", "max_fee_per_aggregate_tx")
        self._min_fee_per_aggregate_tx = ini_file.get_int("transaction", "min_fee_per_aggregate_tx")
        self._expiration_hour = ini_file.get_int("transaction", "expiration_hour")

        self._facade = SymbolFacade(self._network_name)

        # アカウント情報(プロバイダー)
        self._provider_key_pair = KeyPair(PrivateKey(ini_file.get_str("provider_account", "private_key")))
        self._provider_public_key = self._provider_key_pair.public_key
        self._provider_private_key = self._provider_key_pair.private_key
        self._provider_address = self._facade.network.public_key_to_address(self._provider_public_key)

        if self._network_name == "mainnet":
            self._explorer_url = "https://symbol.fyi/transactions/"
        elif self._network_name == "testnet":
            self._explorer_url = "https://testnet.symbol.fyi/transactions/"
        else:
            raise Exception("Unknown network name.")

    def create_nft(
        self,
        file_path: str,
        record_holder_address: str,
        record_holder_name: str,
        record_datetime: datetime,
        record_value: int,
        record_method: str
    ) -> str:
        """
        NFTを作成する

        Parameters
        ----------
        file_path : str
            画像ファイルのパス
        record_holder_address : str
            記録保持者のアドレス(受取人のアドレス)
        record_holder_name : str
            記録保持者の名前(受取人の名前)
        record_datetime : datetime
            測定日時
        record_value : int
            測定結果
        record_method : str
            測定方法

        Returns
        -------
        str
            発行したMosaicID
        """

        # ファイルを読み込む
        file_base64 = File.load_as_base64(file_path)
        file_hash = File.get_file_base64_hash(file_base64)

        mosaic_id = IdGenerator.generate_mosaic_id(self._provider_address, NonceGenerator.generate())

        # メタデータを作成する(NFT概要)
        metadata_overview = self._create_metadata_overview(
            mosaic_id,
            file_hash,
            record_holder_address,
            record_holder_name,
            record_datetime,
            record_value,
            record_method
        )

        # ファイルをメッセージにエンコードする
        messages = NFTDataEncoder.encode_file_base64_to_messages(file_base64, metadata_overview)

        # データトランザクションを作成する
        data_txs = self._create_data_transactions(record_holder_address, messages)
        data_txs_signed = [self._sign_transaction(data_tx, self._provider_key_pair) for data_tx in data_txs]

        # モザイクトランザクションを作成する
        mosaic_tx = self._create_mosaic_transaction(mosaic_id)
        mosaic_tx_signed = self._sign_transaction(mosaic_tx, self._provider_key_pair)

        # メタデータを作成する(参照データ)
        data_tx_hashes = list(map(lambda x: str(x[0]), data_txs_signed))
        metadata_reference_data = self._create_metadata_reference_data(data_tx_hashes)

        # メタデータトランザクションを作成する
        mosaic_meta_tx = self._create_mosaic_metadata_transaction(
            mosaic_id, [metadata_overview] + metadata_reference_data)
        mosaic_meta_tx_signed = self._sign_transaction(mosaic_meta_tx, self._provider_key_pair)

        # モザイクの転送トランザクションを作成する(記録所持者に転送)
        mosaic_trans_tx = self._create_mosaic_transfer_transaction(
            mosaic_id, record_holder_address, json.dumps(metadata_overview["value"])
        )
        mosaic_trans_tx_signed = self._sign_transaction(mosaic_trans_tx, self._provider_key_pair)

        # データトランザクションをアナウンス
        for idx, data_tx_signed in enumerate(data_txs_signed):
            self._logger.info("Announcing data transactions. {}/{}".format(idx+1, len(data_txs_signed)))
            self._announce_transaction(data_tx_signed[0], data_tx_signed[1])
            sleep(3)

        # モザイクトランザクションをアナウンス
        self._logger.info("Announcing mosaic transaction.")
        self._announce_transaction(mosaic_tx_signed[0], mosaic_tx_signed[1])

        # メタデータトランザクションをアナウンス
        self._logger.info("Announcing mosaic metadata transaction.")
        self._announce_transaction(mosaic_meta_tx_signed[0], mosaic_meta_tx_signed[1])

        # モザイクの転送トランザクションをアナウンス
        self._logger.info("Announcing mosaic metadata transaction.")
        self._announce_transaction(mosaic_trans_tx_signed[0], mosaic_trans_tx_signed[1])

        # NFTの作成結果をローカルに保存する
        self._save_nft_summary(
            IdConverter.decimal_int_to_hex_str(mosaic_id),
            file_base64,
            metadata_overview,
            metadata_reference_data,
            str(mosaic_tx_signed[0]),
            str(mosaic_meta_tx_signed[0]),
            str(mosaic_trans_tx_signed[0])
        )

        return IdConverter.decimal_int_to_hex_str(mosaic_id)

    def _get_deadline(self):
        deadline = (int((datetime.today() + timedelta(hours=self._expiration_hour)
                         ).timestamp()) - self._epoch_adjustment) * 1000
        return deadline

    def _create_metadata_overview(
        self,
        mosaic_id: int,
        file_hash: str,
        record_holder_address: str,
        record_holder_name: str,
        record_datetime: datetime,
        record_value: int,
        record_method: str
    ) -> Dict:
        """
        メタデータを作成する(NFT概要)

        media types
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """

        record_datetime_str = format(
            record_datetime.replace(tzinfo=timezone(timedelta(hours=9), "JST")),
            "%Y-%m-%d %H:%M%z(%Z)"
        )

        overview = {
            "description": "NFT概要",
            "key": "overview",
            "value": {
                "version": "speedfighter-nft-1.0",          # NFTのバージョン
                "type": "NFT",                              # 発行種別
                "mosaic_id": IdConverter.decimal_int_to_hex_str(mosaic_id),     # 紐付いたMosaic Id
                "media_type": "image/jpeg",                 # ファイルのMedia Type
                "file_extension": "jpg",                    # ファイルの拡張子
                "file_hash": file_hash,                     # ファイルのチェックサム(SHA256)
                "record_holder_address": record_holder_address,     # 記録保持者のアドレス
                "record_holder_name": record_holder_name,           # 記録保持者の名前
                "record_datetime": record_datetime_str,             # 測定日時
                "record_value": str(record_value),                  # 測定結果
                "record_method": record_method,                     # 測定方法
            }
        }

        return overview

    def _create_metadata_reference_data(self, transaction_hashes: List[str]) -> List[Dict]:
        """
        メタデータを作成する(参照データ)
        """

        key_values = []
        key_values.append({
            "description": "アグリゲートトランザクションの総数",
            "key": "data_length",
            "value": len(transaction_hashes)
        })

        unit = 13
        tx_hashes_groups = [transaction_hashes[i:i+unit] for i in range(0, len(transaction_hashes), unit)]
        for idx, tx_hashes in enumerate(tx_hashes_groups):
            key_values.append({
                "description": "アグリゲートトランザクションのハッシュリスト",
                "key": "data" + str(idx),
                "value": tx_hashes
            })

        return key_values

    def _create_data_transactions(self, record_holder_address: str, messages: List[str]) -> List[AggregateCompleteTransaction]:
        """
        「データの実体を作成するトランザクション」を作成する。
        データはtransfer transacionのmessageに分割格納し、aggregate transactionでこれをまとめる。
        """

        deadline = self._get_deadline()

        # インナートランザクションを作成する
        inner_txs: List[EmbeddedTransferTransaction] = []
        for message in messages:
            inner_tx = self._facade.transaction_factory.create_embedded({
                "type": "transfer_transaction",
                "signer_public_key": self._provider_public_key,
                "recipient_address": record_holder_address,
                "message": bytes(1) + message.encode("utf-8")
            })
            inner_txs.append(inner_tx)

        # アグリゲートトランザクションを作成する
        aggregate_txs: List[AggregateCompleteTransaction] = []
        unit = 100
        inner_txs_groups = [inner_txs[i:i+unit] for i in range(0, len(inner_txs), unit)]
        for inner_txs_group in inner_txs_groups:

            aggregate_tx = self._facade.transaction_factory.create({
                "type": "aggregate_complete_transaction",
                "signer_public_key": self._provider_public_key,
                "fee": Amount(self._calc_fee(len(inner_txs_group))),
                "deadline": deadline,
                "transactions_hash": self._facade.hash_embedded_transactions(inner_txs_group),
                "transactions": inner_txs_group
            })
            aggregate_tx.version = 2
            aggregate_txs.append(aggregate_tx)

        return aggregate_txs

    def _create_mosaic_transaction(self, mosaic_id: int) -> AggregateCompleteTransaction:
        """
        「交換可能なトークンを作成するトランザクション」を作成する。
        トークンは転送可能、数量1、不可変であるmosaicで表現する。
        """

        deadline = self._get_deadline()

        # モザイクのプロパティを定義する
        tx1: EmbeddedMosaicDefinitionTransaction = self._facade.transaction_factory.create_embedded({
            "type": "mosaic_definition_transaction",
            "signer_public_key": self._provider_public_key,
            "id": mosaic_id,
            "duration": BlockDuration(0),
            "nonce": MosaicNonce(NonceGenerator.generate()),
            "flags": MosaicFlags.TRANSFERABLE,
            "divisibility": 0
        })

        # モザイクの供給量を変更する
        tx2: EmbeddedMosaicSupplyChangeTransaction = self._facade.transaction_factory.create_embedded({
            "type": "mosaic_supply_change_transaction",
            "signer_public_key": self._provider_public_key,
            "mosaic_id": mosaic_id,
            "delta": Amount(1),
            "action": MosaicSupplyChangeAction.INCREASE
        })

        # アグリゲートトランザクションを作成する
        aggregate_tx: AggregateCompleteTransaction = self._facade.transaction_factory.create({
            "type": "aggregate_complete_transaction",
            "signer_public_key": self._provider_public_key,
            "fee": Amount(self._calc_fee(3)),
            "deadline": deadline,
            "transactions_hash": self._facade.hash_embedded_transactions([tx1, tx2]),
            "transactions": [tx1, tx2]
        })
        aggregate_tx.version = 2

        return aggregate_tx

    def _create_mosaic_metadata_transaction(self, mosaic_id: int, metadata_key_values: List[Dict]) -> AggregateCompleteTransaction:
        """
        「データの実体とトークンを紐づけるメタデータを作成するトランザクション」を作成する。
        """

        deadline = self._get_deadline()

        # インナートランザクションを作成する
        inner_txs: List[EmbeddedMosaicMetadataTransaction] = []
        for metadata_key_value in metadata_key_values:
            # インナートランザクションを作成する
            inner_tx: EmbeddedMosaicMetadataTransaction = self._facade.transaction_factory.create_embedded({
                "type": "mosaic_metadata_transaction",
                "signer_public_key": self._provider_public_key,
                "target_address": self._provider_address,
                "target_mosaic_id": mosaic_id,
                "scoped_metadata_key": KeyGenerator.generate_uint64_key(metadata_key_value["key"]),
                "value": json.dumps(metadata_key_value["value"]),
                "value_size_delta": len(json.dumps(metadata_key_value["value"]))
            })
            inner_txs.append(inner_tx)

        # アグリゲートトランザクションを作成する
        aggregate_tx: AggregateCompleteTransaction = self._facade.transaction_factory.create({
            "type": "aggregate_complete_transaction",
            "signer_public_key": self._provider_public_key,
            "fee": Amount(self._calc_fee(len(inner_txs))),
            "deadline": deadline,
            "transactions_hash":  self._facade.hash_embedded_transactions(inner_txs),
            "transactions": inner_txs
        })
        aggregate_tx.version = 2

        return aggregate_tx

    def _create_mosaic_transfer_transaction(self, mosaic_id: int, record_holder_address: Address, message: str) -> TransferTransaction:
        """
        モザイクの転送トランザクションを作成する。(記録保持者のアドレスに転送)
        """

        deadline = self._get_deadline()

        mosaics = [{"mosaic_id": mosaic_id, "amount": int(1)}]

        tx: TransferTransaction = self._facade.transaction_factory.create({
            "type": "transfer_transaction",
            "signer_public_key": self._provider_public_key,
            "deadline": deadline,
            "fee": Amount(self._calc_fee(1)),
            "recipient_address": record_holder_address,
            "mosaics": mosaics,
            # NOTE: additional 0 byte at the beginning is added for compatibility with explorer
            # and other tools that treat messages starting with 00 byte as "plain text"
            "message": bytes(1) + message.encode("utf8")
        })

        return tx

    def _sign_transaction(self, transaction, key_pair: KeyPair) -> Tuple[str, bytes]:
        """
        トランザクションに署名する
        """
        signature = self._facade.sign_transaction(key_pair, transaction)
        payload = self._facade.transaction_factory.attach_signature(transaction, signature).encode("utf-8")
        tx_hash = self._facade.hash_transaction(transaction)
        return (tx_hash, payload)

    def _announce_transaction(self, tx_hash: str, payload: bytes) -> str:
        """
        ノードにアナウンスする
        """
        url = self._node_url + "/transactions"
        http_headers = {"Content-type": "application/json"}
        response = requests.put(url, headers=http_headers, data=payload)
        if response.status_code != 202:
            raise Exception("status code is {}".format(response.status_code))

        self._logger.info("tx hash:" + str(tx_hash))
        self._logger.info("status code:" + str(response.status_code))
        self._logger.info(self._explorer_url + str(tx_hash))

        return str(tx_hash)

    def _save_nft_summary(
        self,
        mosaic_id: str,
        file_base64: str,
        metadata_overview: dict,
        metadata_reference_data: dict,
        mosaic_tx: str,
        mosaic_meta_tx: str,
        mosaic_trans_tx: str
    ):
        """
        NFTの作成結果をローカルに保存する
        """

        # コンソール出力
        self._logger.info("NFT Summary")
        self._logger.info("Mosaic ID: {}".format(mosaic_id))
        self._logger.info("Metadata Overview: {}".format(json.dumps(metadata_overview)))
        self._logger.info("Metadata Reference Data Transactions: {}".format(json.dumps(metadata_reference_data)))

        # 保存先ディレクトリ作成
        result_dir_path = Path.join(
            self.project_root_dir_path,
            "assets/nft/result/{:%y%m%d_%H%M%S}_{}".format(dt.datetime.now(), mosaic_id)
        )
        Directory.create_directory(result_dir_path)

        # 画像ファイルを保存
        File.save_file_base64(Path.join(result_dir_path, "nft_data.jpg"), file_base64)

        # サマリ情報を保存
        JsonFile.save(
            Path.join(result_dir_path, "summary.json"),
            {
                "mosaic_id": mosaic_id,
                "metadata_overview": metadata_overview,
                "data_transactions": metadata_reference_data,
                "mosaic_transaction": mosaic_tx,
                "mosaic_metadata_transaction": mosaic_meta_tx,
                "mosaic_transfer_transaction": mosaic_trans_tx
            }
        )

    def _calc_fee(self, tx_count: int) -> int:
        """
        トランザクションの手数料を計算する
        """
        tx_fee = self._max_fee_per_tx * tx_count
        if tx_fee < self._min_fee_per_aggregate_tx:
            tx_fee = self._min_fee_per_aggregate_tx
        if tx_fee > self._max_fee_per_aggregate_tx:
            tx_fee = self._max_fee_per_aggregate_tx
        self._logger.info(f"Calculate fee. tx_count:{tx_count}, tx_fee:{tx_fee}")
        return tx_fee
