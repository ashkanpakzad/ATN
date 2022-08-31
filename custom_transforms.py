# Basic library of transforms not included in torch.torchvision
# By Ashkan Pakzad (ashkanpakzad.github.io) 2022

import torch
import torchvision.transforms.functional as TF
import numpy as np
import numbers
from typing import Tuple, List, Optional
from collections.abc import Sequence


class GaussianNoise(object):
    """Add gaussian noise to image.
    inpired by torchio.

    Args:
        p (float): Desired probability of applying transform
    """

    def __init__(self, p=1.1, mean=(0, 1), std=(0, 0.25)):
        assert isinstance(p, float)
        assert isinstance(mean, (float, tuple))
        assert mean[1] >= mean[0]
        _check_sequence_input(std, "std", req_sizes=(2,))
        assert std[1] >= std[0]

        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        tensor = sample["image"]
        if torch.rand(1) < self.p:
            # get std and mean from random distribution
            std = torch.rand(1) * (self.std[1] - self.std[0]) + self.std[0]
            mean = torch.rand(1) * (self.mean[1] - self.mean[0]) + self.mean[0]
            # add gaussian noise
            tensor = tensor + torch.randn(tensor.size()) * std + mean
            sample["image"] = tensor
        return sample


class GaussianBlur(object):
    """
    Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
        p (float): Desired probability of applying transform

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size=(3, 3), sigma=(0.1, 2.0)):
        self.kernel_size = _setup_size(
            kernel_size, "Kernel size should be a tuple/list of two integers"
        )

        for ks in kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError(
                    "Kernel size value should be an odd and positive number."
                )

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError(
                    "If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError(
                    "sigma values should be positive and of the form (min, max)."
                )
        else:
            raise ValueError(
                "sigma should be a single number or a list/tuple with length 2."
            )

        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        tensor = sample["image"]
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        tensor = TF.gaussian_blur(tensor, self.kernel_size, [sigma, sigma])
        sample["image"] = tensor
        return sample

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)


class ScaleInOut(object):
    """Scale desired range of intensity values of image to desired output desired range. Clips
    beyond range.
    inpired by torchio.

    Args:
        p (float): Desired probability of applying transform
    """

    def __init__(self, inz=(None, None), outz=(-1, 1)):
        assert isinstance(inz, tuple)
        if inz[0] is not None or inz[1] is not None:
            assert inz[1] >= inz[0]
        assert isinstance(outz, (float or int, tuple))
        assert outz[1] >= outz[0]

        self.inz = inz
        self.outz = outz

    def __call__(self, sample):
        tensor = sample["image"]
        # if in is not provided, set to min and max of img
        inz = list(self.inz)
        outz = self.outz
        if self.inz[0] is None:
            inz[0] = tensor.min()
        if self.inz[1] is None:
            inz[1] = tensor.max()
        # linear rescale image as desired
        tensor = (tensor - inz[0]) * \
            ((outz[1] - outz[0]) / (inz[1] - inz[0])) + outz[0]
        # clip intensities to desired range
        tensor[tensor < outz[0]] = outz[0]
        tensor[tensor > outz[1]] = outz[1]
        sample["image"] = tensor
        return sample


class RandomScaling(object):
    """Random affine transformation of the image keeping center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Adapted from pytorch. ONLY intended for use with image only dataset.

    Args:
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        scale,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=0,
    ):

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")

        self.scale = float(torch.empty(1).uniform_(scale[0], scale[1]).item())
        self.resample = self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    def __call__(self, sample):
        tensor = sample["image"]

        fill = self.fill
        if isinstance(tensor, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * TF.get_image_num_channels(tensor)
            else:
                fill = [float(f) for f in fill]

        tensor = TF.affine(
            tensor,
            scale=self.scale,
            interpolation=self.interpolation,
            fill=fill,
            angle=0,
            translate=[0, 0],
            shear=[0, 0],
        )
        sample["image"] = tensor

        return sample


def _check_sequence_input(x, name, req_sizes):
    msg = (
        req_sizes[0] if len(req_sizes) < 2 else " or ".join(
            [str(s) for s in req_sizes])
    )
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be sequence of length {msg}.")


class RandomFlip(object):
    """Random image flip in specified axis.

    Args:
        p (float): Desired probability of applying transform
    """

    def __init__(self, axis, p=1.1):
        assert axis == 0 or axis == 1, f"axis must be either 0 or 1, got {axis}"
        assert isinstance(p, float)

        self.p = p
        # bump up from channel
        self.axis = axis + 1

    def __call__(self, sample):
        if torch.rand(1) < self.p:
            sample["image"] = sample["image"].flip(self.axis)
            if "measures" in sample:
                measures = sample["measures"]
                vals = measures[2:9]
                # flip centre point axis sign
                if self.axis == 1:
                    Cflip = 3
                elif self.axis == 2:
                    Cflip = 2
                vals[Cflip] = -vals[Cflip]
                # flip rotation angle
                vals[4] = -vals[4]
                measures[2:9] = vals
                sample["measures"] = measures

        return sample


class CenterCrop(object):
    """crop to centre by given values

    Args:
    p (float): Desired probability of applying transform
    """

    def __init__(self, outputsize):
        assert (
            len(outputsize) == 2
        ), f"please provide crop size H x W only as list of length 2, got {outputsize}"
        assert all(
            [x > 0 for x in outputsize]
        ), f"crop sizes must be positive, got {outputsize}"
        assert [
            isinstance(x, int) for x in outputsize
        ], f"crop sizes must be integer values, got {outputsize}"

        self.outputsize = outputsize

    def __call__(self, sample):
        tensor = sample["image"]
        tensor = TF.center_crop(tensor, self.outputsize)
        assert list(tensor.size()[1:3]) == list(
            self.outputsize
        ), f"output shape should be the same as input, got {list(tensor.size()[1:3])} and {list(self.outputsize)}"
        sample["image"] = tensor

        return sample


class Normalize(object):
    """Normalize/standardise

    Args:
    p (float): Desired probability of applying transform
    """

    def __init__(self, mean, std):

        self.mean = mean
        self.std = std

    def __call__(self, sample):
        tensor = sample["image"]
        TF.normalize(tensor, self.mean, self.std, inplace=True)
        sample["image"] = tensor
        return sample


class Identity(object):
    """a transform that returns the original sample"""

    def __init__(self):
        pass

    def __call__(self, sample):
        return sample
