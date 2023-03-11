import json
from typing import Dict

import requests


class CatapultRESTAPI:
    """
    REST Gateway
    https://docs.symbol.dev/api.html
    Catapult REST Endpoints (1.0.2)
    https://symbol.github.io/symbol-openapi/v1.0.2/
    """

    def __init__(self, node_url: str) -> None:
        self._node_url = node_url

    def get_epoch_adjustment(self) -> int:
        """
        epochAdjustmentを取得する
        """
        url = self._node_url + "/network/properties"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("status code is {}".format(response.status_code))
        contents = json.loads(response.text)
        epoch_adjustment = int(contents["network"]["epochAdjustment"].replace("s", ""))
        return epoch_adjustment

    def get_currency_mosaic_id(self) -> int:
        """
        currencyMosaicIdを取得する
        """
        url = self._node_url + "/network/properties"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("status code is {}".format(response.status_code))
        contents = json.loads(response.text)
        currency_mosaic_id = int(contents["chain"]["currencyMosaicId"].replace("'", ""), 16)
        return currency_mosaic_id

    def get_mosaic_info(self, mosaic_id: str) -> Dict:
        """
        Get mosaic information
        """
        url = self._node_url + "/mosaics/" + mosaic_id
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("status code is {}".format(response.status_code))
        return json.loads(response.text)

    def get_mosaic_metadata(self, mosaic_id: str) -> Dict:
        """
        Get mosaic metadata
        """
        url = self._node_url + "/metadata"
        params = {
            "targetId": mosaic_id,
            "metadataType": 1
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception("status code is {}".format(response.status_code))
        return json.loads(response.text)

    def get_confirmed_transaction_info(self, transaction_id: str) -> Dict:
        """
        Get confirmed transaction information
        """
        url = self._node_url + "/transactions/confirmed/" + transaction_id
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("status code is {}".format(response.status_code))
        return json.loads(response.text)

    def get_fee_info(self) -> Dict:
        """
        Get fee information
        """
        url = self._node_url + "/network/fees/transaction"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("status code is {}".format(response.status_code))
        return json.loads(response.text)