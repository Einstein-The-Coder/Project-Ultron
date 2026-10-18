import numpy as np


def prepare_input(data):
    """Validate and reshape one 8x8 digit image for the model."""
    try:
        values = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Input must contain numeric pixel values.") from exc

    if values.size != 64:
        raise ValueError("Expected exactly 64 input values.")

    if not np.isfinite(values).all():
        raise ValueError("Input values must all be finite.")

    if np.any(values < 0) or np.any(values > 16):
        raise ValueError("Pixel values must be between 0 and 16.")

    return values.reshape(1, 64)
