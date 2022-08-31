# By Ashkan Pakzad (ashkanpakzad.github.io) 2022

import torch
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from dataset import da_vector_2_angle


def calc_acc(output, type):
    assert type in [0, 1]
    # TARGET 0 FOR REAL AND 1 FOR REFINED
    target = torch.zeros(output.size(0), dtype=torch.long, device=output.device)
    target[:] = type
    # GET SOFTMAX OF OUTPUT
    softmax_output = torch.nn.functional.softmax(output, dim=1)
    acc = softmax_output.max(1)[1] == target
    return acc.sum().div(len(acc))


def getdevice(args):
    if args.devicename is not None:
        devicename = args.devicename
    elif torch.cuda.is_available() and not args.disablecuda:
        devicename = "cuda:0"
    else:
        print("CUDA not available or disabled, using CPU.")
        devicename = "cpu"
    device = torch.device(devicename)
    print(f"Device: {device}")

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    return device


def getnextbatch(iterable, dataloader):
    try:
        batch = next(iterable)
    except StopIteration:
        iterable = iter(dataloader)
        batch = next(iterable)
    return batch, iterable


def getnormjson(filename):
    # get mean and std stats from dataset json file
    with open(filename) as f:
        dict = json.load(f)
    return dict["mean"], dict["std"]


def savesimGANcheckpoint(
    refiner, discriminator, Ref_Opt, Dis_Opt, step, modelv, savepath, mode=None
):
    torch.save(
        {
            "refiner_state_dict": refiner.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "refopt_state_dict": Ref_Opt.state_dict(),
            "discopt_state_dict": Dis_Opt.state_dict(),
            "trainsteps": step,
            "modelv": modelv,
            "mode": mode,
        },
        savepath,
    )


def saveCNRcheckpoint(model, Opt, epoch, modelv, savepath):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "opt_state_dict": Opt.state_dict(),
            "epoch": epoch,
            "modelv": modelv,
        },
        savepath,
    )


def showellipse(img, vals, ax=None, txt=False, pxsize=0.5):
    """
    Plot inner and outer airway wall ellipse given a case.

    Args:
        img: image as ndarray
        vals: measures as ndarray
        davec: bool, true if using double angle representation.
        ax: axis to plot to
        txt: bool. print measurements as mm
        pxsize: pixel size, default is 0.5

    Returns:

    """
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_aspect("equal")
    if isinstance(vals, torch.Tensor):
        vals = vals.numpy()
    if len(vals) == 12:
        # not batch prepared, raw from dataset.
        vals = vals[2:9]
        davec = False
    if len(vals) == 7:
        # before conversion to double angle
        davec = False
    if len(vals) == 8:
        # output from model, converted to double angle
        davec = True

    img = img.squeeze()

    a = vals[0]
    b = vals[1]
    cc_off = vals[2:4]
    if davec is True:
        # [a, b, x0, y0, Wa, Wb, theta1, theta2]
        # convert from double angle representation to radians to degrees
        Wa = vals[4]
        Wb = vals[5]
        ra_ang = da_vector_2_angle(np.expand_dims(vals[6:8], axis=0))
    else:
        # [a, b, x0, y0, theta, Wa, Wb]
        ra_ang = vals[4]
        Wa = vals[5]
        Wb = vals[6]

    ax.imshow(img, cmap="gray")
    # image centre
    imcc = np.array(img.shape) / 2 - pxsize
    # ellipse centre
    cc = imcc + cc_off / pxsize
    ang = np.degrees(ra_ang)
    # function takes width and height of ellipse therefore x2
    # divide by mm size of pixels
    ell1 = Ellipse(
        cc, 2 * vals[0] / pxsize, 2 * vals[1] / pxsize, angle=ang, ec="red", fill=False
    )
    ell2 = Ellipse(
        cc, 2 * Wa / pxsize, 2 * Wb / pxsize, angle=ang, ec="blue", fill=False
    )
    ax.add_patch(ell1)
    ax.add_patch(ell2)
    inner_radius = np.sqrt(vals[0] ** 2 + vals[1] ** 2)
    thickness = np.sqrt(vals[4] ** 2 + vals[5] ** 2) - inner_radius

    if txt:
        print(
            f"0.5mm per pixel. \nInner radius = {inner_radius}mm. \nWall thickness = {thickness}mm"
        )

    return inner_radius, thickness


def noutn(args):
    """Get the number of out nodes for the model dependent arguments set."""
    if args.mode == "ellipse":
        return 8
    elif args.mode == "circle":
        return 2
    else:
        raise ValueError
