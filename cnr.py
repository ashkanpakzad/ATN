# By Ashkan Pakzad (ashkanpakzad.github.io) 2022

import wandb
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import random
import argparse

import engine
import dataset
import model
import util

torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def args_parser():
    parser = argparse.ArgumentParser("Set CNR training", add_help=False)

    # config
    parser.add_argument(
        "--mode",
        default="ellipse",
        help="valid: [measures, ellipse]. Measures learns only radii and wall"
        "wall thicness. ellipse learns inner lumen paramets that an ellipse"
        "fits plus wall thickness.",
    )
    parser.add_argument(
        "--acc_freq",
        default=20,
        type=int,
        help="frequency to perform accessory tasks. Including, run validation dataset for regressor "
        "head, log val stats, save preview images and save model checkpoint",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--resume",
        default=None,
        help="resume from checkpoint, name of checkpoint in experiment dir",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--disabledebug",
        action="store_true",
        help="Disable debug APIs to speed up training.",
    )

    # training hyperparams
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument(
        "--epochs", default=300, type=int, help="Number of epochs to train over"
    )
    parser.add_argument(
        "--train_portion",
        default=0.8,
        help="Portion of data to assign to training, remainder to validation",
    )

    # dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="path to input dataset. If refiner specified, "
        "this should be a synthetic dataset.",
    )
    parser.add_argument(
        "--refiner", type=str, default=None, help="path to refiner model to load"
    )
    parser.add_argument(
        "--inputsize", nargs="+", type=int, default=(32, 32, 1), help="input size in 3D"
    )
    parser.add_argument(
        "--needpreprocess",
        action="store_true",
        help="if data is not already normalised etc.",
    )
    parser.add_argument(
        "--json",
        default=None,
        type=str,
        help="path to json file containing nrm and std info",
    )

    # Model parameters
    parser.add_argument(
        "--transfer", type=str, help="checkpoint to load model for transfer learning"
    )
    parser.add_argument(
        "--modelv", type=int, default=0, help="model version? 0: original"
    )

    # GPU/Cuda management
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="if running on shell cluster set this arg",
    )
    parser.add_argument(
        "--disablecuda",
        action="store_true",
        help="Disable CUDA/GPU use and defer to CPU",
    )
    parser.add_argument(
        "--devicename", default=None, type=str, help="Use a specific device."
    )

    return parser


def main(args):
    if args.disabledebug:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
    # interpret setup args
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = util.getdevice(args)

    data_path = Path(args.dataset)
    if args.json:
        data_path_json = Path(args.json)
    else:
        data_path_json = data_path.with_suffix(".json")

    # CNR model set up
    cnrmodel = model.getCNRmodel(args.modelv, args.mode, device)

    # load json header
    assert (
        data_path_json.exists()
    ), "Need to make json headerfile for dataset, use MakeDatasetHead.py"
    data_stats = util.getnormjson(data_path_json)

    # data loading
    if args.refiner:  # refiner model refines synthetic images on the fly.
        # load refiner checkpoint
        resumepath = Path(args.refiner)
        checkpoint = torch.load(resumepath)
        refiner_modelv = checkpoint["modelv"]
        # create refiner model object
        refiner, _ = model.getmodels(refiner_modelv, device)
        # load model
        refiner.load_state_dict(checkpoint["refiner_state_dict"])
        refiner.to(device)
        engine.setmodel(refiner, False)
    else:
        refiner = None

    base_tsfm = dataset.DeclareTransforms(args.inputsize[0:2])
    if args.needpreprocess:  # assumes data is in HU
        noise_std = (0, 20)
    else:  # assumes data is float, e.g. raw output from a trained refiner
        noise_std = (0, 0.1)

    train_tsfm = base_tsfm(
        data_stats[0],
        data_stats[1],
        noisestd=noise_std,
        gauss_sig=(0.01, 0.5),
        flip=False,
    )
    val_tsfm = base_tsfm(
        data_stats[0],
        data_stats[1],
        noisestd=noise_std,
        gauss_sig=(0.01, 0.5),
        flip=False,
    )

    train_dataset = dataset.ImageData(data_path, transform=train_tsfm)
    val_dataset = dataset.ImageData(data_path, transform=val_tsfm)

    # split dataset
    train_size = float(args.train_portion)
    Nsample = len(train_dataset)
    shuffledidx = torch.randperm(Nsample)
    trainsplit = int(torch.floor(torch.tensor(train_size * Nsample)))
    trainidx = shuffledidx[:trainsplit]
    validx = shuffledidx[trainsplit:]

    train_dataset = Subset(train_dataset, indices=trainidx)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("training dataloader length %d" % len(train_dataloader))

    val_dataset = Subset(val_dataset, indices=validx)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("val dataloader length %d" % len(val_dataloader))
    # loss and optimiser
    Loss = torch.nn.MSELoss(reduction="mean")
    Opt = torch.optim.Adam(
        params=cnrmodel.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
    )

    # wandb
    wandb.init(project="CNR", anonymous="allow")
    wandb.config.update(args)
    wandb.watch(cnrmodel)
    exp_dir = Path(wandb.run.dir)

    ##========================= TRAINING =========================##

    print("Training ...\n")
    engine.CNRtrain(
        args,
        0,
        train_dataloader,
        val_dataloader,
        refiner,
        cnrmodel,
        Opt,
        Loss,
        device,
        exp_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "CNR CT airway measurement", parents=[args_parser()]
    )
    args = parser.parse_args()

    print(args)
    main(args)
