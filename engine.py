# By Ashkan Pakzad (ashkanpakzad.github.io) 2022

import matplotlib
import matplotlib.pyplot as plt
import torch
import enum
from tqdm import tqdm
import numpy as np
import util
from sklearn import metrics
from dataset import prepare_cnr_batch
from pathlib import Path
import wandb


class Action(enum.Enum):
    TRAIN = "Training"
    VALIDATE = "Validation"


class ModelAction(enum.Enum):
    RTRAIN = "RefTraining"
    DTRAIN = "DisTraining"


class TypeAction(enum.Enum):
    PRETRAIN = "Pretrain"
    ADVTRAIN = "Advtrain"


def setmodel(model, istraining):
    if istraining:
        model.train()
    else:
        model.eval()
    for p in model.parameters():
        p.requires_grad = istraining


def prer_train(args, syn_batch, real_batch, refiner, Ref_Opt, SelfRegLoss):

    # set up models and optimisers
    Ref_Opt.zero_grad()

    # run through refiner
    ref_batch = refiner(syn_batch)

    # get losses
    L1_unreg = SelfRegLoss(ref_batch, syn_batch)
    L1_reg = torch.mul(L1_unreg, args.regfactor)
    r_loss = L1_reg

    # update opt and model weights
    r_loss.backward()
    Ref_Opt.step()

    return r_loss


def pred_train(
    args, device, real_batch, syn_batch, refiner, discriminator, Dis_Opt, Adv_Loss
):
    d_loss = None
    d_loss_real = None
    d_loss_ref = None
    acc_real = None
    acc_ref = None

    # set up models and optimisers
    isReftraining = False
    isDistraining = True
    Dis_Opt.zero_grad()

    # get refined batch from refiner and run on discriminator
    ref_batch = refiner(syn_batch)
    d_ref_pred, target_ref = RunDiscriminator(ref_batch, discriminator, 1, device)
    d_loss_ref = Adv_Loss(d_ref_pred, target_ref)
    acc_ref = util.calc_acc(d_ref_pred, 1)
    # run real images on discriminator
    d_real_pred, target_real = RunDiscriminator(real_batch, discriminator, 0, device)
    d_loss_real = Adv_Loss(d_real_pred, target_real)
    acc_real = util.calc_acc(d_real_pred, 0)

    # update discriminator opt and model weights
    d_loss = d_loss_real + d_loss_ref
    d_loss.backward()
    Dis_Opt.step()

    results_dict = {
        "d_loss": d_loss,
        "d_loss_real": d_loss_real,
        "d_loss_ref": d_loss_ref,
        "acc_real": acc_real,
        "acc_ref": acc_ref,
    }
    return results_dict


def adv_train(
    args,
    device,
    action,
    real_batch,
    syn_batch,
    refiner,
    discriminator,
    Ref_Opt,
    Dis_Opt,
    SelfRegLoss,
    Adv_Loss,
    image_history_buffer,
):
    # Either training the refiner or the discriminator not both at the same time.
    r_loss = None
    L1_reg = None
    r_loss_adv = None
    acc_radv = None
    d_loss = None
    d_loss_real = None
    d_loss_ref = None
    acc_real = None
    acc_ref = None

    # set up models and optimisers
    isReftraining = action == action.RTRAIN
    isDistraining = action == action.DTRAIN
    if isReftraining:
        Ref_Opt.zero_grad()
    if isDistraining:
        Dis_Opt.zero_grad()

    # run refiner
    ref_batch = refiner(syn_batch)

    # if training discriminator, use and update ref images history buffer
    if isDistraining:
        half_batch_from_image_history = (
            image_history_buffer.get_from_image_history_buffer(
                nb_to_get=len(ref_batch) // 2
            )
        )
        image_history_buffer.add_to_image_history_buffer(
            ref_batch, nb_to_add=len(ref_batch) // 2
        )
        if len(half_batch_from_image_history):
            ref_batch[: len(ref_batch) // 2] = half_batch_from_image_history

    # run ref images on discriminator
    d_ref_pred, target_ref = RunDiscriminator(ref_batch, discriminator, 1, device)
    d_loss_ref = Adv_Loss(d_ref_pred, target_ref)
    acc_ref = util.calc_acc(d_ref_pred, 1)

    # run real images on discriminator
    if isDistraining and real_batch is not None:
        d_real_pred, target_real = RunDiscriminator(
            real_batch, discriminator, 0, device
        )
        d_loss_real = Adv_Loss(d_real_pred, target_real)
        acc_real = util.calc_acc(d_real_pred, 0)

    # update weights and optimiser
    if isReftraining:
        L1_unreg = SelfRegLoss(ref_batch, syn_batch)
        L1_reg = torch.mul(L1_unreg, args.regfactor)
        r_loss_self = L1_reg

        # add adversarial loss
        zerotensor = torch.zeros(d_ref_pred.size(), dtype=torch.float, device=device)
        zerotensor[:, 0] = 1
        r_loss_adv = Adv_Loss(d_ref_pred, zerotensor)
        r_loss = r_loss_self + r_loss_adv
        acc_radv = util.calc_acc(d_ref_pred, 0)

        r_loss.backward()
        Ref_Opt.step()

    if isDistraining:
        d_loss = d_loss_real + d_loss_ref

        d_loss.backward()
        Dis_Opt.step()

    results_dict = {
        "r_loss": r_loss,
        "r_loss_L1reg": L1_reg,
        "r_loss_adv": r_loss_adv,
        "acc_radv": acc_radv,
        "d_loss": d_loss,
        "d_loss_real": d_loss_real,
        "d_loss_ref": d_loss_ref,
        "acc_real": acc_real,
        "acc_ref": acc_ref,
    }

    return results_dict, ref_batch


def RunDiscriminator(batch, discriminator, targetval, device):
    pred = discriminator(batch).view(-1, 2)
    target = torch.zeros(pred.size(), dtype=torch.float, device=device)
    target[:, targetval] = 1
    return pred, target


def atn_train(
    args, device, real_batch, syn_batch, refiner, Ref_Opt, SelfRegLoss, VGGPL
):

    # set up models and optimisers
    Ref_Opt.zero_grad()

    # run refiner
    ref_batch = refiner(syn_batch)

    # update weights and optimiser
    L1_unreg = SelfRegLoss(ref_batch, syn_batch)
    L1_reg = torch.mul(L1_unreg, args.regfactor)
    PL_reg = torch.mul(VGGPL(ref_batch, syn_batch, real_batch), args.PLfactor)
    r_loss = L1_reg + PL_reg

    # add adversarial loss
    r_loss.backward()
    Ref_Opt.step()

    results_dict = {
        "r_loss": r_loss,
        "r_loss_L1reg": L1_reg,
        "r_loss_pl": PL_reg,
    }

    return results_dict, ref_batch


def CNRepoch(
    args,
    epoch_idx,
    action,
    loader,
    refiner,
    cnr,
    optimizer,
    loss,
    device,
    outn,
    exp_dir,
):
    is_training = action == Action.TRAIN
    epoch_losses = 0
    batch_size = args.batch_size
    setmodel(cnr, is_training)
    all_logits = np.zeros((batch_size * len(loader), outn))
    all_targets = np.zeros((batch_size * len(loader), outn))

    for ii, batch in enumerate(loader):
        inputs, targets = prepare_cnr_batch(args, batch, device, refiner=refiner)
        optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = cnr(inputs)
            batch_losses = loss(logits, targets)
            if is_training:
                batch_losses.backward()
                optimizer.step()
            all_logits[ii * batch_size : ii * batch_size + batch_size, :] = (
                logits.cpu().detach().numpy().tolist()
            )
            all_targets[ii * batch_size : ii * batch_size + batch_size, :] = (
                targets.cpu().detach().numpy().tolist()
            )
            epoch_losses += batch_losses.item()

    if epoch_idx % args.acc_freq == 0 and len(loader) > 0:
        if batch_size < 8:
            nsamples = batch_size
        else:
            nsamples = 8
        action = "train" if is_training else "val"
        matplotlib.use("Agg")
        sampim = inputs[0:nsamples, ...].cpu().numpy()
        samptar = targets[0:nsamples, ...].cpu().numpy()
        samppre = logits[0:nsamples, ...].cpu().detach().numpy()
        fig, ax = plt.subplots(2, nsamples, figsize=(14, 4))
        for ii in range(nsamples):
            if args.mode == "measures":
                util.showimgcirc(sampim[ii], samptar[ii], ax=ax[0][ii])
                util.showimgcirc(sampim[ii], samppre[ii], ax=ax[1][ii])
            elif args.mode == "ellipse":
                util.showellipse(sampim[ii], samptar[ii], ax=ax[0][ii])
                util.showellipse(sampim[ii], samppre[ii], ax=ax[1][ii])
            ax[0][ii].axis("off")
            ax[1][ii].axis("off")
        savename = Path(exp_dir) / f"prev_{action}_{epoch_idx}.png"
        fig.suptitle(f"prev_{action}_{epoch_idx}. Target; Prediction", fontsize=16)
        plt.savefig(savename)
        plt.close(fig)
        caption = f"preview_epoch:{epoch_idx}; preview_{action}; 1st row target, 2nd row prediction"
        images = wandb.Image(str(savename), caption=caption)
        wandb.log({f"{action}_preview": images}, commit=False)

    return epoch_losses, all_targets, all_logits


def CNRtrain(
    args,
    start_epoch,
    training_loader,
    validation_loader,
    refiner,
    cnrmodel,
    optimizer,
    loss,
    device,
    exp_dir,
):
    outn = util.noutn(args)

    num_epochs = args.epochs

    train_all_stats = []
    val_all_stats = []
    lowest_val_loss = np.inf
    for epoch_idx in tqdm(range(start_epoch, num_epochs)):
        train_epoch_result = CNRepoch(
            args,
            epoch_idx,
            Action.TRAIN,
            training_loader,
            refiner,
            cnrmodel,
            optimizer,
            loss,
            device,
            outn,
            exp_dir,
        )

        val_epoch_result = CNRepoch(
            args,
            epoch_idx,
            Action.VALIDATE,
            validation_loader,
            refiner,
            cnrmodel,
            optimizer,
            loss,
            device,
            outn,
            exp_dir,
        )

        train_stats = getstats(
            train_epoch_result[0],
            train_epoch_result[1],
            train_epoch_result[2],
            epoch_idx=epoch_idx,
        )
        val_stats = getstats(
            val_epoch_result[0],
            val_epoch_result[1],
            val_epoch_result[2],
            epoch_idx=epoch_idx,
        )

        train_all_stats.append(train_stats)
        val_all_stats.append(val_stats)

        # logging
        wandb.log(
            {
                "train": {
                    "loss": train_epoch_result[0],
                    "MSE": train_stats["loss"],
                    "MaxError": train_stats["max"].item(),
                    "R^2": train_stats["r^2"].item(),
                    "ExplainedVar": train_stats["explained_var"].item(),
                },
                "val": {
                    "loss": val_epoch_result[0],
                    "MSE": val_stats["loss"],
                    "MaxError": val_stats["max"].item(),
                    "R^2": val_stats["r^2"].item(),
                    "ExplainedVar": val_stats["explained_var"].item(),
                },
            },
            commit=False,
        )

        # scatter graph
        # reduce number to plot for optimised performance
        if epoch_idx % args.acc_freq == 0:
            maxred = 300
            red = args.batch_size if maxred > args.batch_size else maxred
            x_values = val_epoch_result[1][0:red]  # targets
            y_values = val_epoch_result[2][0:red]  # pred
            for ii in range(x_values.shape[1]):
                data = [[x, y] for (x, y) in zip(x_values[:, ii], y_values[:, ii])]
                table = wandb.Table(data=data, columns=["x", "y"])
                scatterplot = wandb.plot.scatter(
                    table, "x", "y", title=f"regr_out_{ii}"
                )
                r2val = metrics.r2_score(x_values[:, ii], y_values[:, ii])
                wandb.log(
                    {f"regr_out_{ii}": scatterplot, f"regr_out_r^2_{ii}": r2val.item()},
                    commit=False,
                )

        # SAVING CHECKPOINTS
        if val_stats["loss"] < lowest_val_loss:  # save the best model
            lowest_val_loss = val_stats["loss"]
            util.saveCNRcheckpoint(
                cnrmodel,
                optimizer,
                epoch_idx,
                args.modelv,
                Path(exp_dir) / "best.tar",
            )
            wandb.log(
                {"best": {"epoch": epoch_idx, "valloss": lowest_val_loss}},
                commit=False,
            )

        wandb.log({}, commit=True)

    return train_all_stats, val_all_stats


def getstats(loss, true, pred, epoch_idx=None, destandardised=False, dataset=None):
    true_mean = true.mean(axis=1)
    pred_mean = pred.mean(axis=1)
    mse = metrics.mean_squared_error(true_mean, pred_mean)
    mae = metrics.mean_absolute_error(true_mean, pred_mean)
    max = metrics.max_error(true_mean, pred_mean)
    r2 = metrics.r2_score(true_mean, pred_mean)
    exvar = metrics.explained_variance_score(true_mean, pred_mean)
    #
    statsdict = {
        "epoch": epoch_idx,
        "loss": loss,
        "mse": mse,
        "mae": mae,
        "max": max,
        "r^2": r2,
        "explained_var": exvar,
        "raw": {
            "losses": loss,
            "true": true,
            "pred": pred,
        },
        "mean": {
            "true": true_mean,
            "pred": pred_mean,
        },
    }

    return statsdict
    return statsdict
