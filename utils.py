import torch
import numpy as np
from torch.nn import ConstantPad3d
from typing import Union, Tuple, Optional, Callable, List


def patches_sampler(input: torch.Tensor, patch_size: Union[int, Tuple[int, int, int]]) -> torch.Tensor:
    """
    Extract patches across a whole volume.
    Args:
        input: image volume of dimensions B1[spatial_dims]
        patch_size: size of pathes, (w, h, d)
    """
    assert input.ndim == 5
    if isinstance(patch_size, int):
        patch_size = [patch_size] * 3
    batch_size = input.shape[0]
    w, h, d = input[0][0].size()
    w_r = -w % patch_size[0]
    h_r = -h % patch_size[1]
    d_r = -d % patch_size[2]
    pad_dims = (d_r // 2, d_r - d_r // 2, h_r, h_r - h_r // 2, w_r, w_r - w_r // 2)
    cpad = ConstantPad3d(pad_dims, value=0)
    input = cpad(input).squeeze(1)
    patches = input.unfold(1, patch_size[0], patch_size[0]) \
        .unfold(2, patch_size[1], patch_size[1]) \
        .unfold(3, patch_size[2], patch_size[2])
    patches = patches.reshape([-1, 1, *patch_size])
    # print(patches.size())
    return patches


def one_hot(labels: torch.Tensor, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    if labels.dim() <= 0:
        raise AssertionError("labels should have dim of 1 or more.")

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    # Transform labels into [0, 1, ..., N-1]
    labels_list = torch.unique(labels).int()
    num_classes = len(labels_list)
    in_lut = torch.zeros(torch.max(labels_list) + 1, dtype=torch.float32)
    for i, lab in enumerate(labels_list):
        in_lut[lab] = i
    labels = in_lut[labels.long()]

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


def one_hot_l(labels: torch.Tensor, labels_list: torch.Tensor, dtype: torch.dtype = torch.float,
              dim: int = 1) -> torch.Tensor:
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    if labels.dim() <= 0:
        raise AssertionError("labels should have dim of 1 or more.")

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    labels = labels.long()

    # Transform labels into [0, 1, ..., N-1]
    # labels_list = torch.unique(labels).int()
    labels_list = labels_list.int()
    labels = torch.where(torch.stack([labels == lab for lab in labels_list]).sum(0).bool(), labels, 0)
    num_classes = len(labels_list)
    in_lut = torch.zeros(torch.max(labels_list) + 1, dtype=torch.float32)
    for i, lab in enumerate(labels_list):
        in_lut[lab] = i
    labels = in_lut[labels.long()]

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


def inout_onehot(source: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    For a tensor `source` and a tensor 'target' of dimensions B1[spatial_dims],
    return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    source_oh = one_hot(source)
    target_oh = one_hot(target)
    in_labels = torch.unique(source).int()
    tar_labels = torch.unique(target).int()
    if (in_labels.shape != tar_labels.shape) or (in_labels != tar_labels).any():
        # raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        labels_list = torch.tensor(
            np.intersect1d(in_labels.cpu().numpy(), tar_labels.cpu().numpy(), assume_unique=True))
        source_oh = one_hot_l(source, labels_list)
        target_oh = one_hot_l(target, labels_list)
        assert source_oh.shape == target_oh.shape
    b, c, h, w, d = source_oh.shape
    source_oh = source_oh.view(-1, h, w, d)
    target_oh = target_oh.view(-1, h, w, d)
    input_vol = source_oh.sum(dim=(1, 2, 3))
    target_vol = target_oh.sum(dim=(1, 2, 3))
    # print(f"vol: input:{input_vol.squeeze()}, target:{target_vol.squeeze()}")
    # ignore incomplete vertebra labels
    mask = torch.logical_and(input_vol > 500, target_vol > 500)
    source_oh = source_oh[mask]
    target_oh = target_oh[mask]
    return source_oh.view(b, -1, h, w, d), target_oh.view(b, -1, h, w, d)
