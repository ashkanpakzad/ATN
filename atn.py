# By Ashkan Pakzad (ashkanpakzad.github.io) 2022
import os
from pathlib import Path
import wandb
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
import random
import argparse
from dataset import DeclareTransforms, prepare_batch, ImageData
from loss import VGGPerceptualLoss
from model import getmodels
from engine import setmodel, atn_train
import util
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def args_parser():
    parser = argparse.ArgumentParser("Set atn", add_help=False)

    # config
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--disabledebug",
        action="store_true",
        help="Disable debug APIs to speed up training.",
    )
    parser.add_argument(
        "--acc_freq",
        default=20,
        type=int,
        help="frequency to perform accessory tasks. Including, run validation"
        " dataset for regressor head, log val stats, save preview images and "
        "save model checkpoint",
    )

    # training hyperparams
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--norm_loss", action="store_true", help="Normalise losses")
    parser.add_argument(
        "--regfactor",
        default=0.01,
        type=float,
        help="regularisation factor on L1 loss",
    )
    parser.add_argument(
        "--PLfactor",
        default=1,
        type=float,
        help="regularisation factor for perceptual losses.",
    )
    parser.add_argument(
        "--PL_style",
        nargs="+",
        default=(0, 1),
        type=int,
        help="VGG layers for style transfer loss.",
    )
    parser.add_argument(
        "--PL_features",
        nargs="+",
        default=(2,),
        type=int,
        help="VGG layers for feature loss.",
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument(
        "--steps",
        default=10000,
        type=int,
        help="Number of steps for training",
    )

    # dataset parameters
    parser.add_argument("--real_dataset", default="")
    parser.add_argument("--synth_dataset", default="")
    parser.add_argument(
        "--inputsize", nargs="+", type=int, default=(32, 32, 1), help="input size in 3D"
    )
    parser.add_argument(
        "--real_scaling",
        nargs="+",
        type=float,
        default=(0.75, 1.25),
        help="random scaling interval to apply to real data input",
    )
    parser.add_argument(
        "--real_noisestd",
        nargs="+",
        type=float,
        default=(0.0, 20.0),
        help="random additive noise gauss std interval to apply to real",
    )
    parser.add_argument(
        "--real_gauss_sig",
        nargs="+",
        type=float,
        default=(0.01, 0.5),
        help="random gaussian blur sigma interval to apply to real",
    )
    parser.add_argument(
        "--train_portion",
        type=float,
        default=1.0,
        help="Portion of data to assign to training",
    )
    # Model parameters
    parser.add_argument(
        "--modelv",
        type=int,
        default=0,
        help="model version? 0: original;",
    )

    # GPU/Cuda management
    parser.add_argument(
        "--disablecuda",
        action="store_true",
        help="Disable CUDA/GPU use and defer to CPU",
    )
    parser.add_argument(
        "--devicename", default=None, type=str, help="Use a specific device."
    )
    parser.add_argument(
        "--anon", action="store_true", help="Force anonymous run on wandb"
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
    synth_data_path = Path(args.synth_dataset)
    synth_data_path_json = synth_data_path.with_suffix(".json")
    real_data_path = Path(args.real_dataset)
    real_data_path_json = real_data_path.with_suffix(".json")

    # model set up
    refiner, _ = getmodels(args.modelv, device)

    # load json header
    assert synth_data_path_json.exists(), "Synthetic json headerfile does not exist"
    assert real_data_path_json.exists(), "Real json headerfile does not exist"
    synth_stats = util.getnormjson(synth_data_path_json)
    real_stats = util.getnormjson(real_data_path_json)

    # data loading
    base_tsfm = DeclareTransforms(args.inputsize[0:2])
    synth_tsfm = base_tsfm(
        synth_stats[0],
        synth_stats[1],
        noisestd=(25, 25),
        gauss_sig=(0.5, 0.875),
        flip=True,
    )
    real_tsfm = base_tsfm(
        real_stats[0],
        real_stats[1],
        noisestd=args.real_noisestd,
        gauss_sig=args.real_gauss_sig,
        flip=True,
        scaling=args.real_scaling,
    )

    # split train dataset for evaluation
    synth_dataset = ImageData(synth_data_path, transform=synth_tsfm)
    real_dataset = ImageData(real_data_path, transform=real_tsfm)
    if args.train_portion < 1:
        train_size = float(args.train_portion)
        Nsample = len(real_dataset)
        shuffledidx = torch.randperm(Nsample)
        trainsplit = int(torch.floor(torch.tensor(train_size * Nsample)))
        trainidx = shuffledidx[:trainsplit]

        real_dataset = Subset(real_dataset, indices=trainidx)

    synth_dataloader = DataLoader(
        synth_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"synth_train_folder {len(synth_dataloader)}")
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("real_train_folder %d" % len(real_dataloader))

    # loss and optimiser
    Ref_Opt = torch.optim.Adam(
        params=refiner.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
    )
    SelfRegLoss = nn.L1Loss(reduction="sum")  # using identity mapping
    if args.PLfactor > 0:
        VGGPL = VGGPerceptualLoss(
            resize=False, style_layers=args.PL_style, feature_layers=args.PL_features
        ).to(device)
    else:
        VGGPL = None

    # wandb
    wandb.config.update(args)
    wandb.watch(refiner)
    exp_dir = Path(wandb.run.dir)
    print(f"Run dir:{str(exp_dir)}")

    ##========================= TRAINING =========================##

    real_iter = iter(real_dataloader)
    synth_iter = iter(synth_dataloader)

    for step in tqdm(range(0, args.steps)):
        setmodel(refiner, True)

        real_iterout, real_iter = util.getnextbatch(real_iter, real_dataloader)
        synth_iterout, synth_iter = util.getnextbatch(synth_iter, synth_dataloader)
        real_batch = prepare_batch(real_iterout, device)
        synth_batch = prepare_batch(synth_iterout, device)

        trainres, ref_batch = atn_train(
            args,
            device,
            real_batch,
            synth_batch,
            refiner,
            Ref_Opt,
            SelfRegLoss,
            VGGPL,
        )

        total_r_loss = trainres["r_loss"] / args.batch_size
        total_r_loss_L1reg = trainres["r_loss_L1reg"] / args.batch_size
        total_r_PLreg = trainres["r_loss_pl"] / args.batch_size

        # if not using PLloss then set to avoid error.
        if total_r_PLreg == 0:
            total_r_PLreg = torch.tensor(0)

        wandb.log(
            {
                "RDstep": step,
                "refiner": {
                    "loss": {
                        "L1reg": total_r_loss_L1reg.item(),
                        "PLreg": total_r_PLreg.item(),
                        "total": total_r_loss.item(),
                    },
                },
            },
            commit=False,
        )

        if step % args.acc_freq == 0:
            realgrid = torchvision.utils.make_grid(real_batch[0:8, ...], normalize=True)
            syngrid = torchvision.utils.make_grid(synth_batch[0:8, ...], normalize=True)
            refgrid = torchvision.utils.make_grid(ref_batch[0:8, ...], normalize=True)

            previewgrid = torch.cat((realgrid, syngrid, refgrid), 1)
            caption = f"preview_step:{step}; Real; Synthetic; Refined"
            images = wandb.Image(previewgrid, caption=caption)
            wandb.log({"preview": images}, commit=False)
            torchvision.utils.save_image(
                previewgrid, Path(exp_dir) / f"preview_{step}.png"
            )

            savepath = Path(exp_dir) / f"checkpoint_{step}.tar"
            torch.save(
                {
                    "refiner_state_dict": refiner.state_dict(),
                    "refopt_state_dict": Ref_Opt.state_dict(),
                    "trainsteps": step,
                    "modelv": args.modelv,
                },
                savepath,
            )

        wandb.log({}, commit=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "ATN for refining synthetic CT airway patches=", parents=[args_parser()]
    )
    args = parser.parse_args()

    # init wandb
    if args.anon:
        anon = "true"
    else:
        anon = "allow"
    wandb.init(project="ATN", anonymous="must")

    print(args)
    main(args)
