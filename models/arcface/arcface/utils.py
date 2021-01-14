import numpy as np
import tensorflow as tf
from absl import logging


def l2_normalize(x, axis=1):
    """l2 normalization"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output
