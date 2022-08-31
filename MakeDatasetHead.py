# By Ashkan Pakzad (ashkanpakzad.github.io) 2022

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import custom_transforms

import argparse
from dataset import ImageData
import json
from tqdm import tqdm


def args_parser():
    parser = argparse.ArgumentParser('Explore', add_help=False)
    parser.add_argument('datadir', type=str,
                        help='path to dir holding the dataset')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Number of images to load each time to calculate values, default is fullsize')
    parser.add_argument('--inputsize', nargs='+', type=int, default=(32, 32),
                        help='tuple, size of images to calc stats on, default is 32x32, set to None if not desired')
    parser.add_argument('--nosave', action='store_true',
                        help='Do not save stats to file, just display')
    return parser


def main(args):
    path = Path(args.datadir)
    assert path.is_dir(
    ), f'dataset datadir is not a directory, got {path} instead.'
    datsetname = path.stem

    # Get all images
    if args.inputsize:
        img_tsfm = custom_transforms.CenterCrop(list(args.inputsize))
    else:
        img_tsfm = custom_transforms.Identity()

    dataset = ImageData(path, transform=img_tsfm)

    if args.batch_size:
        batchsize = args.batch_size
    else:
        batchsize = len(dataset)
    loader = DataLoader(dataset, batch_size=batchsize, drop_last=False)

    # Calculate total N
    totalN = len(dataset)
    print(f'Total number of images: {totalN}')

    # Compute mean and std
    runmean = torch.tensor(0.0)
    runvar = torch.tensor(0.0)
    runn = torch.tensor(0.0)
    # See https://stats.stackexchange.com/a/435643 for adding means and stds equations
    for i, batch in (enumerate(tqdm(loader))):
        inp = batch['image']
        newn = inp.shape[0]
        newmean = inp.flatten().mean()
        newvar = inp.flatten().var()

        runmean = (runn*runmean + newn*newmean) / (runn + newn)
        runvar = ((runn - 1)*runvar + (newn - 1)*newvar + (runn*newn/(runn + newn))
                  * (runmean**2+newmean**2 - 2*runmean*newmean)) / (runn + newn - 1)
        runn = runn + newn

    allmean = runmean
    allstd = torch.sqrt(runvar)
    print(
        f'Dataset: {datsetname} \n Final (mean, std): ({allmean.item()}, {allstd.item()})')

    # Create JSON object
    if not args.nosave:
        outdict = {
            'name': datsetname,
            'len': totalN,
            'mean': allmean.item(),
            'std': allstd.item(),
        }

        outputpath = path.parent / (str(datsetname) + '.json')

        # save to json
        with open(outputpath, 'w', encoding='utf-8') as f:
            json.dump(outdict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Create header JSON file for dataset in directory', parents=[args_parser()])
    args = parser.parse_args()
    main(args)
