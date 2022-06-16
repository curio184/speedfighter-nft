from datetime import datetime

from speedfighter.nft.nft_creator import NFTCreator
from speedfighter.nft.nft_data_restorer import NFTDataRestorer


class NFTUsecase:

    @staticmethod
    def create_nft():
        """
        NFTを発行する
        """

        # NFTに記載する情報
        file_path = "./assets/new_record.jpg"
        record_holder_address = "TA3HQR6NPMXK7W6EP3AO6X5S4OSHVBU3ZEWBTNQ"
        record_holder_name = "y.oya"
        record_datetime = datetime.now()
        record_value = 94
        record_method = "Bushnell 101911 Velocity Speed Gun"

        # NFTを発行する
        creator = NFTCreator()
        mosaic_id = creator.create_nft(
            file_path,
            record_holder_address,
            record_holder_name,
            record_datetime,
            record_value,
            record_method
        )

        # 発行したモザイクID
        print(mosaic_id)

    @staticmethod
    def restore_nft_data():
        """
        NFTデータをSymbolチェーンから復元する
        """

        # NFTデータをSymbolチェーンから復元する
        restorer = NFTDataRestorer()
        restorer.restore_nft_data("35AB044006D7010B", "./assets/restore_data.jpg")


if __name__ == "__main__":

    # NFT発行機能のみをテストできるスクリプト
    # あらかじめconfigs/symbol.iniを設定しておく

    # NFTを発行する
    NFTUsecase.create_nft()

    # NFTデータをSymbolチェーンから復元する
    NFTUsecase.restore_nft_data()
