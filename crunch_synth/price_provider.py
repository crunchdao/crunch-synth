from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union, overload

import pandas
import requests


class PriceUnavailableError(ValueError):
    """
    Raised when the price provider cannot fetch the price for an asset.
    """


HistoryColumns = Literal[
    "open",
    "high",
    "low",
    "close",
    "volume",
    "number_of_trades",
]


class PriceDbClient:

    _HISTORY_URL = "https://pricedb.crunchdao.com/v1/prices"

    def _do_get_price_history(
        self,
        *,
        asset: str,
        from_: datetime,
        to: datetime,
        columns: List[HistoryColumns],
        timeout: int = 30,
    ) -> Dict[str, List[Any]]:
        query: Dict[str, Any] = {
            "asset": asset,
            "from": from_.isoformat(),
            "to": to.isoformat(),
            "columns": columns,
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

        return root

    @overload
    def get_price_history(
        self,
        *,
        asset: str,
        from_: datetime,
        to: datetime,
        timeout: int = 30,
    ) -> List[tuple[datetime, float]]:
        ...

    @overload
    def get_price_history(
        self,
        *,
        asset: str,
        from_: datetime,
        to: datetime,
        columns: List[HistoryColumns],
        timeout: int = 30,
    ) -> pandas.DataFrame:
        ...

    def get_price_history(
        self,
        *,
        asset: str,
        from_: datetime,
        to: datetime,
        columns: Optional[List[HistoryColumns]] = None,
        timeout: int = 30,
    ) -> Union[List[tuple[datetime, float]], pandas.DataFrame]:
        if columns is not None and not len(columns):
            raise ValueError("columns list cannot be empty")

        root = self._do_get_price_history(
            asset=asset,
            from_=from_,
            to=to,
            columns=columns or ["close"],
            timeout=timeout,
        )

        if columns is None:
            return list(zip(root["timestamp"], root["close"]))

        else:
            dataframe = pandas.DataFrame(root)

            dataframe["timestamp"] = pandas.to_datetime(dataframe["timestamp"], unit="s")

            missing_columns = set(columns) - set(dataframe.columns)
            for column in missing_columns:
                dataframe[column] = float("nan")

            existing_columns = set(columns) & set(dataframe.columns)
            for column in existing_columns:
                dataframe[column] = pandas.to_numeric(dataframe[column])  # handle conversion from None to NaN

            return dataframe


pricedb = PriceDbClient()
