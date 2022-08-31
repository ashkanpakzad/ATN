import torch
from torch.utils.data import DataLoader
from dataset import DeclareTransforms, ImageData
from model import getmodels, getCNRmodel
import util
import numpy as np

from pathlib import Path

def outputellipse(vals):
    '''
    Convert output of model variables to interpretable ellipse parameters.
    input vals structure must be [Ra, Rb, Cx, Cy, Wa, Wb, Ta, Tb]
    R = inner ellipse
    W = outer ellipse
    T = double angle vector representation of rotation
    C = centre offset
    Output will be [Ra, Rb, Cx, Cy, Wa, Wb, T]
    T = single angle representation in radians
    '''
    # convert from double angle representation to radians to degrees
    ra_ang = util.da_vector_2_angle(vals[:,6:8])
    vals[:,6] = ra_ang
    vals = np.delete(vals, 7, 1)
    return vals

def runCNRmodel(modelpath, data_path, batch_size, mean, var, gpu=True):
    # setup device
    if torch.cuda.is_available() and gpu is True:
        devicename = "cuda:0"
    else:
        devicename = "cpu"
    device = torch.device(devicename)

    # model config
    checkpoint = torch.load(modelpath)
    if 'modelv' in checkpoint:
        modelv = checkpoint['modelv']
    else:
        modelv = 0
    mode = "ellipse"
    CNRmodel = getCNRmodel(modelv, mode, device)
    CNRmodel.load_state_dict(checkpoint['model_state_dict'])
    CNRmodel.eval()

    # data config
    inputsize = [32,32]
    base_tsfm = DeclareTransforms(inputsize)
    tsfm = base_tsfm(mean, var, noisestd=None, gauss_sig=None, flip=False)

    # load data
    dataset = ImageData(data_path, transform=tsfm)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

    # inference
    outputmeasures = np.zeros((len(dataset), 8))
    count = 0
    for batch in dataloader:
        minidx = count * batch_size
        outbatch = CNRmodel(batch['image'].to(device))
        count += 1

        # store measures
        outputmeasures[minidx:minidx+len(batch['image']), :] = outbatch.detach().cpu().numpy()

    outputvals = outputellipse(outputmeasures)

    return outputvals
