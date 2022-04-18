import numpy as np

from straw.correctors.base import BaseCorrector
from straw.io.params import StreamParams


class BiasCorrector(BaseCorrector):
    def apply(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        bias = np.zeros(samplebuffer.shape[0], dtype=np.int8)
        for i in range(samplebuffer.shape[0]):
            bias[i] = samplebuffer[i].mean()
            samplebuffer[i] -= bias[i]
        params.bias = bias

    def apply_revert(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        for i in range(params.channels):
            samplebuffer[i] += params.bias[i]
