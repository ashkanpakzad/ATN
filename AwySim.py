# By Ashkan Pakzad (ashkanpakzad.github.io) 2022
import numpy as np
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.util import random_noise
import csv


class AirwaySim:
    def __init__(self, rng, rngdict, imsz=[40, 40], pixsz=0.05):
        # rng = numpy random number generator object
        # dictvar = dictionary of numbers that specify distribution for airway patch variables
        # seed = rng seed to use
        self.rng = rng
        self.rngdict = rngdict
        # fixed vals
        self.imsz = imsz  # image size mm
        self.pixsz = pixsz  # superresolution pixel size mm/pixel

    def GenerateVariables(self):
        '''Generates the core variables for creating the airway patch'''
        rngdict = self.rngdict
        dictvar = {}
        dictvar['Lr'], dictvar['Wr'], dictvar['offset'], dictvar['Ae'], dictvar['Li'], dictvar['Wi'] = self.GenAirwayVars()

        ## Parenchyma values
        # higher = more fibrotic appearance.
        dictvar['p_std'] = self.rng.uniform(
            rngdict['p_std'][0], rngdict['p_std'][1])
        dictvar['p_mean'] = (self.rng.standard_normal()
                             * rngdict['p_mean'][1] + rngdict['p_mean'][0])

        ## Vessel values
        dictvar['n_v'] = self.rng.integers(
            0, high=rngdict['n_v'], endpoint=True)  # number of vessels
        dictvar['Vr'] = np.zeros([dictvar['n_v'], ])
        dictvar['Vi'] = np.zeros([dictvar['n_v'], ])
        dictvar['Ve'] = np.zeros([dictvar['n_v'], ])
        dictvar['Vpos'] = np.zeros([dictvar['n_v'], ])
        for i in range(0, dictvar['n_v']):
            dictvar['Vr'][i], dictvar['Vi'][i], dictvar['Ve'][i], dictvar['Vpos'][i] = self.GenVesselVars(
                dictvar['Lr'])
        # others
        dictvar['CTpixsz'] = [0.5, 0.5]

        ## Proximal Bifurcation patch - case for if there is a proximal bifurication
        self.proxbifurcationpatch = (self.rng.uniform(
            0, 1) < rngdict['proxbifurcationpatch'])
        self.dictvar = dictvar

    def GenAirwayVars(self):
        '''Generates characteristics for an airway using the distibution dictionary'''
        rngdict = self.rngdict
        # airway centre offset x,y mm
        Lr = self.rng.uniform(rngdict['Lr'][0], rngdict['Lr'][1])
        # offset should be upper bound by lumen radius and sign maintained
        offset = (self.rng.standard_normal(2)
                  - rngdict['offset'][0]) * rngdict['offset'][1]
        for i in range(len(offset)):
            if abs(offset[i]) > Lr:
                offset[i] = (Lr) * np.sign(offset[i])

        Wr = Lr + self.rng.uniform(rngdict['Wr'][0][0], rngdict['Wr'][0][1]) * Lr + self.rng.uniform(
            rngdict['Wr'][1][0], rngdict['Wr'][1][1])  # wall radius mm
        Ae = self.rng.uniform(rngdict['Ae'][0], rngdict['Ae'][1])
        # lumen intensity HU
        Li = self.rng.uniform(rngdict['Li'][0], rngdict['Li'][1])
        # wall intensity HU
        Wi = self.rng.uniform(rngdict['Wi'][0], rngdict['Wi'][1])
        return Lr, Wr, offset, Ae, Li, Wi

    def GenVesselVars(self, Lr):
        '''Generates characteristics for a vessel using the distibution dictionary'''
        rngdict = self.rngdict
        Vr = self.rng.uniform(Lr, Lr + rngdict['Vr'])  # vessel radius mm
        # vessel intensity HU
        Vi = self.rng.uniform(rngdict['Vi'][0], rngdict['Vi'][1])
        Ve = self.rng.uniform(rngdict['Ve'][0], rngdict['Ve'][1])
        Vpos = self.rng.uniform(rngdict['Vpos'][0], rngdict['Vpos'][1])
        return Vr, Vi, Ve, Vpos

    def GenerateCanvas(self):
        # set up axes
        x = np.arange(0, self.imsz[0], self.pixsz)
        y = np.arange(0, self.imsz[1], self.pixsz)
        X, Y = np.meshgrid(x, y)
        # grid centre
        self.xgridcentreidx = np.round(x.shape[0] / 2)
        self.ygridcentreidx = np.round(y.shape[0] / 2)
        self.xgridcentre = x[int(self.xgridcentreidx)]
        self.ygridcentre = y[int(self.ygridcentreidx)]
        return X, Y, x, y

    def DrawEllipseRot(self, X, Y, xc, yc, r1, r2, p):
        ra = np.max([r1, r2])
        rb = np.min([r1, r2])
        vox = ((((X - xc) * np.cos(p) + (Y - yc) * np.sin(p)) ** 2) / (ra ** 2)) + \
              ((((X - xc) * np.sin(p) - (Y - yc) * np.cos(p)) ** 2) / (rb ** 2)) <= 1
        # a, b, x0, y0, theta
        ell_para = [ra, rb, xc-self.xgridcentre, yc-self.ygridcentre, p]
        return vox, ell_para

    def GenerateSims(self, rng):
        # generate random variables
        self.GenerateVariables()
        X, Y, x, y = self.GenerateCanvas()
        # set up parenchyma noise
        I_p = self.dictvar['p_mean'] + (
                rng.standard_normal((x.shape[0], y.shape[0])) * self.dictvar['p_std'])

        ## add airway lumen and wall
        # set up centre
        xci = int(np.round(self.xgridcentreidx
                           + self.dictvar['offset'][0]/self.pixsz))
        yci = int(np.round(self.ygridcentreidx
                           + self.dictvar['offset'][1]/self.pixsz))

        # airway rotation
        arot = rng.uniform(-np.pi/2, np.pi/2)

        # ellipsoidness between 0 and 1
        Lr1 = self.dictvar['Ae'] * self.dictvar['Lr']
        Lr2 = self.dictvar['Lr']**2 / Lr1
        Wr1 = self.dictvar['Ae'] * self.dictvar['Wr']
        Wr2 = self.dictvar['Wr'] ** 2 / Wr1

        # draw lumen
        lumenvox, ell_para = self.DrawEllipseRot(
            X, Y, x[xci], y[yci], Lr1, Lr2, arot)
        wallvox, Well_para = self.DrawEllipseRot(
            X, Y, x[xci], y[yci], Wr1, Wr2, arot)

        I_a = I_p  # assign parenchyma image
        I_a[wallvox] = self.dictvar['Wi']
        I_a[lumenvox] = self.dictvar['Li']

        ## add adjacent vessels
        I_v = I_a  # assign airway image
        for ii in range(0, self.dictvar['n_v']):
            # identify vessel centres
            vcx = x[xci] + (self.dictvar['Wr'] + self.dictvar['Vr']
                            [ii]) * np.cos(self.dictvar['Vpos'][ii])
            vcy = y[yci] + (self.dictvar['Wr'] + self.dictvar['Vr']
                            [ii]) * np.sin(self.dictvar['Vpos'][ii])

            # vessel rotation
            vrot = rng.uniform(-np.pi/2, np.pi/2)

            # vessel ellipsoidness
            Vr1 = self.dictvar['Ve'][ii] * self.dictvar['Vr'][ii]
            Vr2 = self.dictvar['Vr'][ii] ** 2 / Vr1

            # draw vessels
            vesselvox, _ = self.DrawEllipseRot(X, Y, vcx, vcy, Vr1, Vr2, vrot)
            I_v[vesselvox] = self.dictvar['Vi'][ii]

        # If this patch is a particular case patch, can draw over the vessels but airway
        I_B = I_v  # assign vessel image

        # CASE: proxbifurcationpatch
        # if there is a proximal bifurcation patch, generate a neighbouring airway of same size
        # airway lumen size within mm difference
        if self.proxbifurcationpatch:
            B_diff1 = rng.uniform(-0.25, 0.25)
            B_diff2 = rng.uniform(-0.25, 0.25)
            # offset from centre must be at the two greatest diameters
            offsetmin = np.max([Lr1, Lr2]) + np.max([Lr1+B_diff1, Lr2+B_diff2])
            B_xci = int((xci + offsetmin/self.pixsz)) * RandomSign(rng)
            B_yci = int((yci + offsetmin/self.pixsz)) * RandomSign(rng)
            if B_xci > 799:
                B_xci = 799
            elif B_xci < 0:
                B_xci = 0
            if B_yci > 799:
                B_yci = 799
            elif B_yci < 0:
                B_yci = 0
            # draw bifurcation airway
            Blumenvox, _ = self.DrawEllipseRot(
                X, Y, x[B_xci], y[B_yci], Lr1+B_diff1, Lr2+B_diff2, arot)
            Bwallvox, _ = self.DrawEllipseRot(
                X, Y, x[B_xci], y[B_yci], Wr1+B_diff1, Wr2+B_diff2, arot)
            I_B[Bwallvox] = self.dictvar['Wi']
            I_B[Blumenvox] = self.dictvar['Li']


        # resample pixel size is determined as half min of CT pixel sizes.
        resamp_sz = np.floor(np.min(self.dictvar['CTpixsz']) * 10) / 10

        I_out = DownSample(I_B, self.pixsz, resamp_sz, order=3)

        radii = [self.dictvar['Lr'], self.dictvar['Wr']]

        # a, b, x0, y0, theta
        ell_para.extend(Well_para)
        return (I_out, radii, ell_para)


def DownSample(input, pixsz, resamp_sz, order=3):
    # resize using new pixel size
    dncols = np.floor((pixsz / resamp_sz) * input.shape[0])
    dnrows = np.floor((pixsz / resamp_sz) * input.shape[1])
    I_down = resize(input, (dnrows, dncols), order=order)
    return I_down


class CSVfile:
    def __init__(self, csvpath, datacols):
        self.csvpath = str(csvpath)
        self.datacols = datacols
        # make csv file on disk
        writerow = ['path'] + datacols
        with open(self.csvpath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(writerow)
        # declare cols

    def __call__(self, img_path, data):
        # open csv file to append
        writerow = [str(img_path)] + data
        with open(self.csvpath, 'a', newline='') as f:
            writer = csv.writer(f)
            # append and save row.
            writer.writerow(writerow)


def RandomSign(rng):
    return 1 if rng.uniform() < 0.5 else -1
