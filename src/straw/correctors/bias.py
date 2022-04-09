import numpy as np

from straw.correctors.base import BaseCorrector


class BiasCorrector(BaseCorrector):
    def global_apply(self, samplebuffer: np.ndarray) -> (np.ndarray, np.ndarray):
        bias = np.zeros(samplebuffer.shape[0], dtype=np.int16)
        for i in range(samplebuffer.shape[0]):
            bias[i] = samplebuffer[i].mean()
            samplebuffer[i] -= bias[i]
        return samplebuffer, bias
