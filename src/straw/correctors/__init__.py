import numpy as np

from straw.io.params import StreamParams
from .bias import BiasCorrector
from .decorrelator import Decorrelator
from .gain import GainCorrector
from .shift import ShiftCorrector


def apply_corrections(data: np.array,
                      corrections: tuple,
                      target_params: StreamParams = StreamParams(),
                      force_inplace: bool = False):
    """
    Apply the given corrections to the data in thegiven order
    :param data: array to apply the corrections to - ndarray with dimensions (channels, samples)
    :param corrections: corrections to apply
    :param target_params: stream params where the correction params should be stored
    :param force_inplace: force the operation inplace (mainly shift correction)
    :return:
    """
    for correction in corrections:
        if correction == "gain":
            GainCorrector().apply(data, target_params)
        elif correction == "bias":
            BiasCorrector().apply(data, target_params)
        elif correction == "shift":
            sc = ShiftCorrector()
            sc.apply(data, target_params)
            if force_inplace:
                sc.apply_to_ndarray(data)
        else:
            raise ValueError(f"Invalid correction name: '{correction}', must be one of ('gain', 'bias', 'shift')")
