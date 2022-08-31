# By Ashkan Pakzad (ashkanpakzad.github.io) 2022

import torch
from pathlib import Path
import tifffile
import numpy as np
from torchvision import transforms
import custom_transforms
import pandas as pd
import copy
from math import pi


class DeclareTransforms:
    def __init__(self, inputsize):
        self.inputsize = inputsize

    def __call__(
        self,
        mean,
        std,
        flip=True,
        noisestd=(0, 20),
        gauss_sig=(0.01, 0.5),
        scaling=None,
    ):
        if noisestd is not None:
            addnoise = custom_transforms.GaussianNoise(
                mean=(0, 0), std=noisestd)
        else:
            addnoise = custom_transforms.Identity()

        if gauss_sig is not None:
            blur = custom_transforms.GaussianBlur(
                kernel_size=(3, 3), sigma=gauss_sig)
        else:
            blur = custom_transforms.Identity()

        if flip:
            hflip = custom_transforms.RandomFlip(0, p=0.2)
            vflip = custom_transforms.RandomFlip(1, p=0.2)
        else:
            hflip = custom_transforms.Identity()
            vflip = custom_transforms.Identity()

        if scaling is not None:
            scaleaffine = custom_transforms.RandomScaling(scaling)
        else:
            scaleaffine = custom_transforms.Identity()

        tsfm = transforms.Compose(
            [
                scaleaffine,
                addnoise,
                blur,
                custom_transforms.Normalize((mean,), (std,)),
                custom_transforms.CenterCrop(self.inputsize),
                hflip,
                vflip,
            ]
        )
        return tsfm


def prepare_batch(input, device):
    return input["image"].to(device)


def prepare_cnr_batch(args, input, device, refiner=None):
    images = input["image"].to(device)
    if refiner is not None:  # refine images first
        images = refiner(images)
    if args.mode == "ellipse":
        # Inner ellipse a,b,x0,y0,theta and Wa, Wb
        thetas = input["measures"][:, 6]
        da = angle_2_da_vector(thetas)
        # 2:6 = Ra, Rb, Cx, Cy, Wa, Wb
        vals = torch.cat(
            (input["measures"][:, 2:6], input["measures"][:, 7:9], da), dim=1
        )
    return images, vals.to(device)


class ImageData(torch.utils.data.Dataset):
    def __init__(self, datasetname, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if isinstance(datasetname, str):
            datasetname = Path(datasetname)
        assert datasetname.is_dir()

        # load csv file if it exists for measurements
        csv_file = datasetname.with_suffix(".csv")
        if csv_file.is_file():
            self.frame = pd.read_csv(csv_file)
        else:
            self.frame = None

        self.img_dir = datasetname
        self.transform = transform
        self.image_paths = sorted(self.img_dir.glob("*"))

        # get image format: tif or npy
        self.format = next(self.img_dir.iterdir()).suffix

    def __len__(self):
        if self.frame is not None:
            return len(self.frame)
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}

        if self.frame is not None:
            img_name = self.frame.iloc[idx, 0]
            img_path = self.img_dir / Path(img_name).with_suffix(self.format)
            measures = self.frame.iloc[idx, 1:]
            measures = torch.tensor([measures], dtype=torch.float)
            measures = measures.flatten()
            # convert 2nd measure (outer radius) to wall thickness
            measures[1] = measures[1] - measures[0]
            sample["measures"] = measures
        else:
            img_path = str(self.image_paths[idx])

        if self.format == ".tif" or self.format == ".tiff":
            raw_img = tifffile.imread(img_path)
        elif self.format == ".npy":
            raw_img = np.load(img_path)
        else:
            raise TypeError(
                "image data type must be either int 16 tiff or npy")
        img = torch.tensor(raw_img[None, ...], dtype=torch.float)

        sample["image"] = img
        sample["index"] = idx
        sample["name"] = str(img_path)

        if self.transform:
            sample = self.transform(sample)

        return sample


def EllipseImageSpace(vals, imshape, pxsize=0.5, radout=True):
    """
    Convert ellipse from generalised annotation to given image space.

    vals - list of ellipse parameters, a,b,x0,y0,theta and Wa, Wb
    imshape - list of 2, shape of image
    pxsize - size of pixels in mm
    radout - get output as radians
    Assumes vals is a list in the following order:
    """
    # copy vals for output
    newvals = copy.deepcopy(vals)
    # scale all except angle by pxsize
    ii = np.arange(0, len(vals))
    newvals = [newvals[i] / pxsize if i not in [4] else newvals[i] for i in ii]

    # ellipse centre
    # imcc = np.array(imshape) / 2 - pxsize

    newvals[2:4] = torch.tensor(imshape) / 2 - \
        pxsize + torch.tensor(newvals[2:4])
    # convert from radians to degrees
    if radout:
        ang = vals[4]
    else:
        ang = np.degrees(vals[4])
    newvals[4] = ang
    newvals = torch.stack(newvals)
    return newvals


def angle_2_da_vector(angles: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/KluvaDa/Chromosomes MIT license
    Angles in radians to double-angle vector space; 0 radians -> (1, 0), pi/4 radians -> (0, 1)
    Args:
        angles: torch.Tenor of shape (batch, 1, x, y)
    Returns: torch tensor of shape (batch, 2, x, y)
    """
    double_angle = angles * 2
    da_vectors_x = torch.cos(double_angle)
    da_vectors_y = torch.sin(double_angle)
    da_vectors = torch.stack([da_vectors_x, da_vectors_y], dim=1)
    return da_vectors


def da_vector_2_angle(vectors: np.ndarray) -> np.ndarray:
    """
    https://github.com/KluvaDa/Chromosomes MIT license
    Double-angle vector space to angles in radians in range [0, pi); (1, 0) -> 0 radians, (0, 1) -> pi/4 radians
    Args:
        vectors: torch.Tensor of shape (batch, 2,)
    Returns: torch.Tensor of shape (batch, 1)
    """
    double_angle = np.arctan2(vectors[..., 1], vectors[..., 0])
    double_angle = np.remainder(double_angle, 2 * pi)
    angle = double_angle / 2
    return angle
