from tensorflow.keras import backend as K
import numpy as np

def average(x, per_image=False, class_weights=None, **kwargs):
    if per_image:
        x = K.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)

def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""
    backend = K
    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (3, 0, 1, 2))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    return x

def get_reduce_axes(per_image, **kwargs):
    backend = K
    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes

def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs

def round_if_needed(x, threshold, **kwargs):
    backend = K
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x