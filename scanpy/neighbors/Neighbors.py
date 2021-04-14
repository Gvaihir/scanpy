from typing import Mapping

import numpy as np

from .neighbors_config import method_type, metric

RPForestDict = Mapping[str, Mapping[str, np.ndarray]]

_Method = method_type()
_Metric = metric()
