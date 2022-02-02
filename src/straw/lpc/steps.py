import numpy as np


def predict_signal(frame: np.array, qlp: np.array, order: int, shift: int):
    """
    Executes LPC prediction
    The resulting predicted signal starts with the order-th sample
    :param frame: signal frame
    :param qlp: quantized LPC coefficients
    :param order: LPC order
    :param shift: coefficient quantization shift
    :return: predicted frame with shape [order:]
    """
    if order <= 0 or qlp is None:
        return None

    return np.convolve(frame, qlp, mode="full")[order - 1:-order] >> shift
