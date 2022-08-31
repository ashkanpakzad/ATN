# By Ashkan Pakzad (ashkanpakzad.github.io) 2022
import os
import wandb
from tqdm import tqdm
import util
from imagehistorybuffer import ImageHistoryBuffer
from engine import setmodel, ModelAction, prer_train, pred_train, adv_train
from model import getmodels
from loss import VGGPerceptualLoss
from dataset import DeclareTransforms, prepare_batch, ImageData
import argparse
import random
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch
from datetime import timedelta
from pathlib import Path


torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def args_parser():
    parser = argparse.ArgumentParser("Set GANCNN", add_help=False)

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
        help="frequency to perform accessory tasks. Including, run validation dataset for regressor "
        "head, log val stats, save preview images and save model checkpoint",
    )
    parser.add_argument(
        "--anon", action="store_true", help="Force anonymous run on wandb"
    )
    # training hyperparams
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument(
        "--regfactor",
        default=0.01,
        type=float,
        help="regularisation factor on Refiner L1 loss",
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument(
        "--R_train",
        default=1000,
        type=int,
        help="Number of iteration steps to pretrain Refiner model",
    )
    parser.add_argument(
        "--D_train",
        default=200,
        type=int,
        help="Number of iteration steps to pretrain Discriminator model",
    )
    parser.add_argument(
        "--RD_train",
        default=10000,
        type=int,
        help="Number of steps for adversarial training",
    )
    parser.add_argument(
        "--k_r",
        default=50,
        type=int,
        help="Number of updates on Refiner for each step of adversarial training",
    )
    parser.add_argument(
        "--k_d",
        default=1,
        type=int,
        help="Number of updates on Discriminator for each step of adversarial training",
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

    # Model parameters
    parser.add_argument(
        "--modelv", type=int, default=0, help="model version? 0: original;"
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
    refiner, discriminator = getmodels(args.modelv, device)

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

    synth_dataset = ImageData(synth_data_path, transform=synth_tsfm)
    real_dataset = ImageData(real_data_path, transform=real_tsfm)

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

    # setup history buffer
    image_history_buffer = ImageHistoryBuffer(
        (0, 1, args.inputsize[0], args.inputsize[1]),
        args.batch_size * 10,
        args.batch_size,
        device,
    )

    # loss and optimiser
    Ref_Opt = torch.optim.Adam(
        params=refiner.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
    )
    Dis_Opt = torch.optim.Adam(
        params=discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
    )

    SelfRegLoss = nn.L1Loss(reduction="sum")  # using identity mapping
    Adv_Loss = nn.BCEWithLogitsLoss(reduction="sum")

    # wandb
    wandb.config.update(args)
    wandb.watch(refiner)
    wandb.watch(discriminator)
    exp_dir = Path(wandb.run.dir)

    ##========================= PRETRAINED REFINER =========================##
    print("Pretrain refiner ...")
    setmodel(refiner, True)
    setmodel(discriminator, False)

    synth_iter = iter(synth_dataloader)
    real_iter = iter(real_dataloader)
    for prerstep in tqdm(range(args.R_train)):
        synth_iterout, synth_iter = util.getnextbatch(synth_iter, synth_dataloader)
        synth_batch = prepare_batch(synth_iterout, device)
        real_iterout, real_iter = util.getnextbatch(real_iter, real_dataloader)
        real_batch = prepare_batch(real_iterout, device)

        r_loss_reg = prer_train(
            args, synth_batch, real_batch, refiner, Ref_Opt, SelfRegLoss
        )

        wandb.log(
            {
                "preRstep": prerstep,
                "pretrain": {"refiner": {"r_loss": r_loss_reg.item()}},
            }
        )
    print(f"Save pretrained refiner to {exp_dir}/R_pre.pkl")
    torch.save(refiner.state_dict(), Path(exp_dir) / "R_pre.pkl")

    ##========================= PRETRAINED DISCRIMINATOR =========================##

    print("Pretrain discriminator ...")
    setmodel(refiner, False)
    setmodel(discriminator, True)

    real_iter = iter(real_dataloader)
    synth_iter = iter(synth_dataloader)
    for predstep in tqdm(range(args.D_train)):
        real_iterout, real_iter = util.getnextbatch(real_iter, real_dataloader)
        synth_iterout, synth_iter = util.getnextbatch(synth_iter, synth_dataloader)
        real_batch = prepare_batch(real_iterout, device)
        synth_batch = prepare_batch(synth_iterout, device)

        dpretrainres = pred_train(
            args,
            device,
            real_batch,
            synth_batch,
            refiner,
            discriminator,
            Dis_Opt,
            Adv_Loss,
        )
        wandb.log(
            {
                "preDstep": predstep,
                "pretrain": {
                    "discriminator": {
                        "d_loss_real": dpretrainres["d_loss_real"].item(),
                        "d_loss_ref": dpretrainres["d_loss_ref"].item(),
                        "d_loss_total": dpretrainres["d_loss"].item(),
                        "real": dpretrainres["acc_real"].item(),
                        "ref": dpretrainres["acc_ref"].item(),
                    }
                },
            }
        )

    print(f"Save D_pre to {exp_dir}/D_pre.pkl")
    torch.save(discriminator.state_dict(), Path(exp_dir) / "D_pre.pkl")

    ##========================= ADVERSARIAL TRAINING =========================##
    #

    print("Joint Training ...")
    real_iter = iter(real_dataloader)
    synth_iter = iter(synth_dataloader)

    for step in tqdm(range(0, args.RD_train)):
        # ========= train the Refiner =========
        setmodel(refiner, True)
        setmodel(discriminator, False)
        total_r_loss = 0.0
        total_r_loss_L1reg = 0.0
        total_r_PLreg = 0.0
        total_r_loss_adv = 0.0
        total_acc_adv = 0.0

        for index in range(args.k_r):
            action = ModelAction("RefTraining")
            synth_iterout, synth_iter = util.getnextbatch(synth_iter, synth_dataloader)
            synth_batch = prepare_batch(synth_iterout, device)

            radvtrainres, _ = adv_train(
                args,
                device,
                action,
                real_batch,
                synth_batch,
                refiner,
                discriminator,
                Ref_Opt,
                Dis_Opt,
                SelfRegLoss,
                Adv_Loss,
                image_history_buffer,
            )

            total_r_loss += radvtrainres["r_loss"] / args.batch_size
            total_r_loss_L1reg += radvtrainres["r_loss_L1reg"] / args.batch_size
            total_r_loss_adv += radvtrainres["r_loss_adv"] / args.batch_size
            total_acc_adv += radvtrainres["acc_radv"]
        mean_r_loss = total_r_loss / args.k_r
        mean_r_loss_L1reg = total_r_loss_L1reg / args.k_r
        mean_r_loss_adv = total_r_loss_adv / args.k_r
        mean_acc_adv = total_acc_adv / args.k_r

        # ========= train the Discriminator =========
        setmodel(refiner, False)
        setmodel(discriminator, True)
        total_d_loss_real = 0.0
        total_d_loss_ref = 0.0
        total_d_loss = 0.0
        total_d_accuracy_real = 0.0
        total_d_accuracy_ref = 0.0

        for index in range(args.k_d):
            action = ModelAction("DisTraining")
            real_iterout, real_iter = util.getnextbatch(real_iter, real_dataloader)
            synth_iterout, synth_iter = util.getnextbatch(synth_iter, synth_dataloader)
            real_batch = prepare_batch(real_iterout, device)
            synth_batch = prepare_batch(synth_iterout, device)

            # run d training
            dadvtrainres, ref_batch = adv_train(
                args,
                device,
                action,
                real_batch,
                synth_batch,
                refiner,
                discriminator,
                Ref_Opt,
                Dis_Opt,
                SelfRegLoss,
                Adv_Loss,
                image_history_buffer,
            )

            # accumulate per kd step
            total_d_loss_real += dadvtrainres["d_loss_real"].item() / args.batch_size
            total_d_loss_ref += dadvtrainres["d_loss_ref"].item() / args.batch_size
            total_d_loss += dadvtrainres["d_loss"].item() / args.batch_size
            total_d_accuracy_real += dadvtrainres["acc_real"].item()
            total_d_accuracy_ref += dadvtrainres["acc_ref"].item()

        # compute mean and log
        mean_d_loss_real = total_d_loss_real / args.k_d
        mean_d_loss_ref = total_d_loss_ref / args.k_d
        mean_d_loss = total_d_loss / args.k_d
        mean_d_accuracy_real = total_d_accuracy_real / args.k_d
        mean_d_accuracy_ref = total_d_accuracy_ref / args.k_d

        wandb.log(
            {
                "RDstep": step,
                "refiner": {
                    "loss": {
                        "adv": mean_r_loss_adv.item(),
                        "L1reg": mean_r_loss_L1reg.item(),
                        "total": mean_r_loss.item(),
                    },
                    "accuracy": mean_acc_adv.item(),
                },
                "discriminator": {
                    "loss": {
                        "real": mean_d_loss_real,
                        "ref": mean_d_loss_ref,
                        "total": mean_d_loss,
                    },
                    "accuracy": {
                        "real": mean_d_accuracy_real,
                        "ref": mean_d_accuracy_ref,
                    },
                },
            },
            commit=False,
        )

        if mean_d_accuracy_real < 0.02 or mean_d_accuracy_real > 0.98:
            wandb.alert(
                title="GAN Collapsed",
                text=f"GAN has collapsed. Discriminator Accuracy Real is {mean_d_accuracy_real}",
                level=wandb.AlertLevel.WARN,
                wait_duration=timedelta(minutes=60),
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

            util.savesimGANcheckpoint(
                refiner,
                discriminator,
                Ref_Opt,
                Dis_Opt,
                step,
                args.modelv,
                Path(exp_dir) / f"checkpoint_{step}.tar",
            )

        wandb.log({}, commit=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "simGAN for refining synthetic CT airway patches=", parents=[args_parser()]
    )
    args = parser.parse_args()

    # init wandb
    if args.anon:
        anon = "force"
    else:
        anon = "allow"
    wandb.init(project="simGAN", anonymous=anon)

    print(args)
    main(args)
