import numpy as np

def to_array(image):
    array = np.array(image, dtype=np.float32)[..., :3]
    array = array / 255.
    return array


def l2_normalize(x, axis=0):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / norm


def distance(a, b):
    # Euclidean distance
    # return np.linalg.norm(a - b)
    # Cosine distance, ||a|| and ||b|| is one because embeddings are normalized.
    # No need to compute np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.dot(a, b)
