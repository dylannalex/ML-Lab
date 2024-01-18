import requests
import pandas as pd
from time import sleep


def _get_crypto_prices(symbol, interval, range_period):
    """
    Fetches historical market prices for a specified cryptocurrency.

    Parameters:
    - symbol (str): The symbol or ID of the cryptocurrency (e.g., 'bitcoin').
    - interval (str): The time interval for the data (e.g., 'daily').
    - range_period (int): The number of days for the specified time interval.

    Returns:
    - list: A list of historical prices for the cryptocurrency.
    """
    base_url = "https://api.coingecko.com/api/v3/coins/"
    endpoint = f"{symbol}/market_chart"
    params = {"vs_currency": "usd", "interval": interval, "days": range_period}
    response = requests.get(base_url + endpoint, params=params)
    data = response.json()
    try:
        return data["prices"]
    except KeyError:
        raise Exception(f"Could not extract {symbol} prices: {response.content}")


def _create_dataframe(prices, symbol):
    """
    Creates a Pandas DataFrame from a list of cryptocurrency prices.

    Parameters:
    - prices (list): List of historical prices for a cryptocurrency.
    - symbol (str): The symbol or ID of the cryptocurrency.

    Returns:
    - pd.DataFrame: DataFrame with timestamp as index and cryptocurrency price as a column.
    """
    df = pd.DataFrame(prices, columns=["timestamp", f"{symbol}_price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df


def _right_indexmerge_dataframes(dataframes):
    """
    Merges multiple DataFrames based on their indices.

    Parameters:
    - dataframes (list): List of DataFrames to be merged.

    Returns:
    - pd.DataFrame: Merged DataFrame.
    """
    df_merged = dataframes[0]
    for df in dataframes[1:]:
        df_merged = pd.merge(
            df_merged, df, left_index=True, right_index=True, how="outer"
        )

    return df_merged


def create_prices_dataframe(symbols, interval, range_period, sleep_time=3):
    """
    Fetches and combines cryptocurrency prices for specified symbols.

    Parameters:
    - symbols (list): List of cryptocurrency symbols or IDs (e.g., ["bitcoin", "ethereum"]).
    - interval (str): The time interval for the data (e.g., 'daily').
    - range_period (int): The number of days for the specified time interval.
    - sleep_time (int): The sleep time in seconds between API requests.

    Returns:
    - pd.DataFrame: Combined DataFrame with prices for all specified cryptocurrencies.
    """
    crypto_dfs = []
    for symbol in symbols:
        prices = _get_crypto_prices(symbol, interval, range_period)
        df = _create_dataframe(prices, symbol)
        crypto_dfs.append(df)
        sleep(sleep_time)

    all_crypto_df = _right_indexmerge_dataframes(crypto_dfs)
    return all_crypto_df


def get_supported_coins():
    base_url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(base_url)
    data = response.json()

    coin_names = [coin["name"] for coin in data]
    return coin_names
