from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

"""
based on https://stackoverflow.com/a/53135031
"""


class ParallelCompute:
    cpus = None
    _apply_args = None
    _apply_kwargs = None

    def __init__(self, cpus=cpu_count()):
        """
        Initialize the compute class
        :param cpus: number of CPUs to use when parallel is True, None means use all
        :param args: default function args
        """
        self.cpus = cpus

    def _parallelize(self, data, func):
        data_split = np.array_split(data, self.cpus)
        pool = Pool(self.cpus)
        data = pd.concat(pool.map(func, data_split))
        pool.close()
        pool.join()
        return data

    def _run_on_subset(self, func, data_subset):
        return data_subset.apply(func, args=self._apply_args, **self._apply_kwargs)

    def apply(self, data, func, args=None, **kwargs) -> pd.Series:
        """
        Apply the functiom to the given DataFrame or Series in parallel
        :param data: DataFrame or Series to which func will be applied
        :param func: function to apply
        :param args: args to use in apply
        :param kwargs: kwargs to use in apply
        :return: DataFrame or Series with applied data
        """
        self._apply_args = args
        self._apply_kwargs = kwargs
        return self._parallelize(data, partial(self._run_on_subset, func))
