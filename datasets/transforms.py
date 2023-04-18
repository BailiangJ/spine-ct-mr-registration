from copy import deepcopy
from itertools import chain
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
from monai.config import IndexSelection, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad.array import (BorderPad, CropForeground,
                                            SpatialCrop)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Transform
from monai.transforms.utils import compute_divisible_spatial_size, is_positive
from monai.utils import (Method, NumpyPadMode, PytorchPadMode, ensure_tuple,
                         ensure_tuple_rep, fall_back_tuple)
from monai.utils.enums import InverseKeys
from monai.utils.type_conversion import convert_data_type

NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode,
                             str]
PadModeSequence = Union[Sequence[Union[NumpyPadMode, PytorchPadMode, str]],
                        NumpyPadMode, PytorchPadMode, str]


class CropForegroundd(MapTransform, InvertibleTransform):
    """Dictionary-based version :py:class:`monai.transforms.CropForeground`. Crop only
    the foreground object of the expected images. The typical usage is to help training
    and evaluation if the valid part is small in the whole medical image. The valid part
    can be determined by any field in the data with `source_key`, for example:

    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.
    """
    def __init__(
        self,
        keys: KeysCollection,
        source_key: Union[str, Sequence[str]],
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
        start_coord_key: str = 'foreground_start_coord',
        end_coord_key: str = 'foreground_end_coord',
        allow_missing_keys: bool = False,
        **np_kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            source_key: data source to generate the bounding box of foreground, can be image or label, etc.
            select_fn: function to select expected foreground, default is to select values > 0.
            channel_indices: if defined, select foreground only on the specified channels
                of image. if None, select foreground on the whole image.
            margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
            k_divisible: make each spatial dimension to be divisible by k, default to 1.
                if `k_divisible` is an int, the same `k` be applied to all the input spatial dimensions.
            mode: padding mode {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                one of the listed string values or a user supplied function. Defaults to ``"constant"``.
                see also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                it also can be a sequence of string, each element corresponds to a key in ``keys``.
            start_coord_key: key to record the start coordinate of spatial bounding box for foreground.
            end_coord_key: key to record the end coordinate of spatial bounding box for foreground.
            allow_missing_keys: don't raise exception if key is missing.
            np_kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
                more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html

        """
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.cropper = CropForeground(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            k_divisible=k_divisible,
            **np_kwargs,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.k_divisible = k_divisible

    def __call__(
            self, data: Mapping[Hashable,
                                np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        img: np.ndarray
        box_start_list = []
        box_end_list = []
        for key in self.source_key:
            img, *_ = convert_data_type(d[key], np.ndarray)  # type: ignore
            box_start, box_end = self.cropper.compute_bounding_box(img=img)
            box_start_list.append(box_start)
            box_end_list.append(box_end)
            # print(f"box_start:{box_start}, box_end:{box_end}")
        if len(box_start_list) > 1:
            box_start = np.minimum(*box_start_list)
            box_end = np.maximum(*box_end_list)
            # make spatial size divisible
            orig_spatial_size = box_end - box_start
            spatial_size = np.asarray(
                compute_divisible_spatial_size(spatial_shape=orig_spatial_size,
                                               k=self.k_divisible))
            box_start = box_start - np.floor_divide(
                np.asarray(spatial_size) - orig_spatial_size, 2)
            box_end = box_start + spatial_size
        d[self.start_coord_key] = box_start
        d[self.end_coord_key] = box_end
        # print(box_start, box_end, box_end-box_start)
        for key, m in self.key_iterator(d, self.mode):
            self.push_transform(d,
                                key,
                                extra_info={
                                    'box_start': box_start,
                                    'box_end': box_end
                                })
            d[key] = self.cropper.crop_pad(img=d[key],
                                           box_start=box_start,
                                           box_end=box_end,
                                           mode=m)

        return d

    def inverse(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[InverseKeys.ORIG_SIZE])
            cur_size = np.asarray(d[key].shape[1:])
            extra_info = transform[InverseKeys.EXTRA_INFO]
            box_start = np.asarray(extra_info['box_start'])
            box_end = np.asarray(extra_info['box_end'])
            # first crop the padding part
            roi_start = np.maximum(-box_start, 0)
            roi_end = cur_size - np.maximum(box_end - orig_size, 0)

            d[key] = SpatialCrop(roi_start=roi_start, roi_end=roi_end)(d[key])

            # update bounding box to pad
            pad_to_start = np.maximum(box_start, 0)
            pad_to_end = orig_size - np.minimum(box_end, orig_size)
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            # second pad back the original size
            d[key] = BorderPad(pad)(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class MergeMaskd(MapTransform):
    """The segmentation label mask specified by 'keys' will be merged and stored.

    Args:
            keys: keys of the corresponding items to be merged.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
    """
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        assert len(keys) == 2

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        mask_list = []
        for key in self.key_iterator(d):
            mask_list.append(d[key])

        assert len(mask_list) == 2
        vert_range = np.intersect1d(np.unique(mask_list[0]),
                                    np.unique(mask_list[1]),
                                    assume_unique=True)
        vert_range = np.delete(vert_range, vert_range.argmin())  # background

        mask_list_1 = np.where(np.isin(mask_list[1], vert_range), mask_list[1],
                               0)
        mask_list_0 = np.where(np.isin(mask_list[0], vert_range), mask_list[0],
                               0)
        merge_mask = np.where(mask_list_0 == 0, mask_list_1, 0) + mask_list_0

        # mask_list[1][~np.isin(mask_list[1], vert_range)] = 0
        # merge_mask = np.where(mask_list[0] == 0, mask_list[1],
        #                       0) + mask_list[0]

        d['merge_mask'] = merge_mask
        return d


class OneHotd(MapTransform):
    """The segmentation label mask specified by 'keys' will be transformed to one-hot
    format.

    Args:
            keys: keys of the corresponding items to be merged.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
    """
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        assert len(keys) == 2

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        mask_list = []
        for key in self.key_iterator(d):
            mask_list.append(d[key])
        assert len(mask_list) == 2

        # Get the union of unique values in both segmentation labels
        classes = np.intersect1d(np.unique(mask_list[0]),
                                 np.unique(mask_list[1]),
                                 assume_unique=True)
        num_classes = len(classes)

        # One-hot encode each segmentation mask
        for key in self.key_iterator(d):
            mask = d[key]
            # the first dimmension of mask is channel
            mask = mask.squeeze()
            one_hot = np.zeros((num_classes, *mask.shape), dtype=np.uint8)

            for i, c in enumerate(classes):
                one_hot[i][mask == c] = 1

            # the key is 'mr_mask' or 'ct_mask'
            output_key = ('_').join([key.split('_')[0], 'oh'])
            d[output_key] = one_hot

        return d
