from __future__ import absolute_import

import numpy as np
import torch
import torch.nn.functional as F

_func_conv_nd_table = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def spatial_filter_nd(x: torch.Tensor,
                      kernel: torch.Tensor,
                      mode: str = 'replicate') -> torch.Tensor:
    """N-dimensional spatial filter with padding.

    Args:
        x: the shape should be BNH[WD]
        kernel: Weight tensor (e.g., Gaussain kernel, Gradient kernel).
        mode (str, optional): Padding mode. Defaults to 'replicate'.

    Returns:
        torch.Tensor: Output tensor
    """
    kernel = kernel.to(x)

    n_dim = x.dim() - 2
    if n_dim <= 0 or n_dim > 3:
        raise AssertionError(
            f'the spatial dims of input should be 1, 2 or 3, get{n_dim}')
    conv = _func_conv_nd_table[n_dim]

    pad = [None, None] * n_dim
    pad[0::2] = kernel.shape[2:]
    pad[1::2] = kernel.shape[2:]
    pad = [k // 2 for k in pad]

    return conv(F.pad(x, pad=pad, mode=mode), kernel)


# NOTE: Gaussian kernel
def _gauss_1d(x, mu, sigma):
    return 1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 /
                                                      (2 * sigma**2))


def gauss_kernel_1d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = np.arange(-lw, lw + 1)
    kernel_1d = _gauss_1d(x, 0., sigma)
    return kernel_1d / kernel_1d.sum()


def gauss_kernel_2d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = np.arange(-lw, lw + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    kernel_2d = _gauss_1d(X, 0., sigma) \
                * _gauss_1d(Y, 0., sigma)
    return kernel_2d / kernel_2d.sum()


def gauss_kernel_3d(sigma, truncate=4.0):
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    x = y = z = np.arange(-lw, lw + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    kernel_3d = _gauss_1d(X, 0., sigma) \
                * _gauss_1d(Y, 0., sigma) \
                * _gauss_1d(Z, 0., sigma)
    return kernel_3d / kernel_3d.sum()


# NOTE: Average kernel
def _average_kernel_nd(ndim, kernel_size):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim

    kernel_nd = np.ones(kernel_size)
    kernel_nd /= np.sum(kernel_nd)

    return kernel_nd


def average_kernel_1d(kernel_size):
    return _average_kernel_nd(1, kernel_size)


def average_kernel_2d(kernel_size):
    return _average_kernel_nd(2, kernel_size)


def average_kernel_3d(kernel_size):
    return _average_kernel_nd(3, kernel_size)


# NOTE: Gradient kernel
def gradient_kernel_1d(method='default'):
    if method == 'default':
        kernel_1d = np.array([-1, 0, +1])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return kernel_1d


def gradient_kernel_2d(method='default', axis=0):
    if method == 'default':
        kernel_2d = np.array([[0, -1, 0], [0, 0, 0], [0, +1, 0]])
    elif method == 'sobel':
        kernel_2d = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
    elif method == 'prewitt':
        kernel_2d = np.array([[-1, -1, -1], [0, 0, 0], [+1, +1, +1]])
    elif method == 'isotropic':
        kernel_2d = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0],
                              [+1, +np.sqrt(2), +1]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_2d, 0, axis)


def gradient_kernel_3d(method='default', axis=0):
    if method == 'default':
        kernel_3d = np.array([[[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              [[0, 0, 0], [0, +1, 0], [0, 0, 0]]])
    elif method == 'sobel':
        kernel_3d = np.array([[[-1, -3, -1], [-3, -6, -3], [-1, -3, -1]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              [[+1, +3, +1], [+3, +6, +3], [+1, +3, +1]]])
    elif method == 'prewitt':
        kernel_3d = np.array([[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              [[+1, +1, +1], [+1, +1, +1], [+1, +1, +1]]])
    elif method == 'isotropic':
        kernel_3d = np.array([[[-1, -1, -1], [-1, -np.sqrt(2),
                                              -1], [-1, -1, -1]],
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                              [[+1, +1, +1], [+1, +np.sqrt(2), +1],
                               [+1, +1, +1]]])
    else:
        raise ValueError('unsupported method (got {})'.format(method))

    return np.moveaxis(kernel_3d, 0, axis)
