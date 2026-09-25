import numpy as np


def prepare_input(data):

    data = np.asarray(
        data,
        dtype=np.float32
    )

    if data.size != 64:
        raise ValueError(
            "Ultron expects 64 input values."
        )

    return data.reshape(1, 64)