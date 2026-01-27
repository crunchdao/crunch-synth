from datetime import datetime

import requests


class PriceUnavailableError(ValueError):
    """
    Raised when the price provider cannot fetch the price for an asset.
    """


class PriceDbClient:

    _HISTORY_URL = "https://pricedb.crunchdao.com/v1/prices"

    def get_price_history(
        self,
        *,
        asset: str,
        from_: datetime,
        to: datetime,
        timeout=30,
    ) -> list[tuple[float, int]]:
        query = {
            "asset": asset,
            "from": from_.isoformat(),
            "to": to.isoformat(),
        }

        try:
            response = requests.get(
                self._HISTORY_URL,
                timeout=timeout,
                params=query,
            )

            response.raise_for_status()

            root = response.json()
        except Exception as error:
            raise PriceUnavailableError(f"could not get price history for {asset}: {error}") from error

        return list(zip(root["timestamp"], root["close"]))


pricedb = PriceDbClient()
