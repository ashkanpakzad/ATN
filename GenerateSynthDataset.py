# By Ashkan Pakzad (ashkanpakzad.github.io) 2022

'''
Expected input JSON fields and the random distribution values that they represent.

U: X ~ U(a, b) = uniform distribution with lower and upper limits a and b.
G: X ~ G(m, s) = gaussian/normal distribution with mean and standard deviation m and s.
c: C = constant = a constant value

'name': str # must be different to name of the json file
'prefix': str
'seed': int
'superpx': float in mm per pixel
'imszmm': float final image size in px
###
'p_std': [a, b], # U parenchyma std
'p_mean': [m, s], # G parenchyma mean
# Airway values
'offset': [m, s], # G
'Lr': [shape, scale], # U
'Wr': 2x2, # Lr + U * Lr + U
'Ae': # U airway ellipsoidness out of 1
'Li': [a, b], # U
'Wi': [a, b], # U
# Vessel values
'n_v': b, # U (a = 0 by default)
'Vr': c, # U (Lr, Vr * Lr)
'Ve': # U vessel ellipsoidness out of 1
'Vi': [a, b], # U
'Vpos': [a, b], # U
# CT downsample pixel size
'CTpixsz': [a, b], # U
# smoothing in mm i.e. sigma on filter
'smooth': [a, b], # U
'noise' : c, # quantum noise variance in HU


variables beyond control of JSON files:
rotation of airway and vessel. Uniformly free for axis to land in any way.
'''

import AwySim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
import json
import argparse
import copy


def args_parser():
    parser = argparse.ArgumentParser('Generate airway dataset', add_help=False)
    parser.add_argument(
        'jsonpath', help='jsonpath to json file detailing dataset configuration')
    parser.add_argument('--output_dir', '-o', help='directory to store output')
    parser.add_argument('--N', '-n', type=int,
                        help='number of images to generate, should balance out n real')
    parser.add_argument('--show', action='store_true',
                        help='plot and show outputs')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save output')
    return parser


def main(args):

    # load json file
    with open(args.jsonpath) as f:
        jsondict = json.load(f)

    # set up
    if not args.nosave:
        savedir = Path(args.output_dir)
        savedir.mkdir(parents=True, exist_ok=False)
        csvpath = savedir.with_suffix('.csv')
        datacols = ['innerradius', 'outerradius', 'La', 'Lb',
                    'Lx0', 'Ly0', 'Lp', 'Wa', 'Wb', 'Wx0', 'Wy0', 'Wp']
        datafile = AwySim.CSVfile(csvpath, datacols)
    prefix = jsondict['prefix']

    # initiate random number generator
    rng = np.random.default_rng(jsondict['seed'])
    # initiate airway property generator
    AwySimGen = AwySim.AirwaySim(rng, jsondict)

    for i in tqdm(range(args.N)):
        # generate
        output, radii, ellipsepara = AwySimGen.GenerateSims(rng)
        datarow = copy.copy(radii)
        datarow.extend(ellipsepara)
        intout = output.astype(np.int16)
        if args.show:
            plt.imshow(intout, cmap='gray')
            plt.show()

        # save
        if not args.nosave:
            filename = prefix+str(i)
            savename = savedir / (filename + '.tif')
            tifffile.imwrite(str(savename), intout)
            datafile(filename, datarow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'airway generator', parents=[args_parser()])
    args = parser.parse_args()
    main(args)
