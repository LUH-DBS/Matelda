import html
import logging
import os.path

import numpy as np
import pandas as pd
import re


def value_normalizer(value: str) -> str:
    """
    This method takes a value and minimally normalizes it. (Raha's value normalizer)
    """
    if value is not np.NAN:
        value = html.unescape(value)
        value = re.sub("[\t\n ]+", " ", value, re.UNICODE)
        value = value.strip("\t\n ")
    return value


def read_csv(path: str, low_memory: bool = False) -> pd.DataFrame:
    """
    This method reads a table from a csv file path, with pandas default null values and str data type
    Args:
        low_memory: whether to use low memory mode (bool), default False
        path: table path (str)

    Returns:
        pandas dataframe of the table
    """
    logger = logging.getLogger()
    logger.info("Reading table, name: %s", os.path.basename(path))

    return pd.read_csv(path, sep=",", header="infer", low_memory=low_memory)\
        .applymap(lambda x: value_normalizer(x) if isinstance(x, str) else x)\
        .applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)



