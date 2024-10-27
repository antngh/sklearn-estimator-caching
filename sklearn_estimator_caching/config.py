"""
Config for the sklearn caching module
"""

from typing import Union

import numpy.typing as npt
import pandas as pd
import polars as pl

# Supported data types in sklearn pipelines that can be cached
DataType = Union[npt.NDArray, pd.DataFrame, pd.Series, pl.DataFrame, pl.Series]

# split the polars df into blocks of this many rows when converting to string
# in _EstimatorCachingWrapperMetadataEncoder
BLOCK_SIZE = int(1e6)
