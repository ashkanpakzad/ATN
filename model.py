# By Ashkan Pakzad (ashkanpakzad.github.io) 2022


from torch import nn
import util


def parsemode(mode):
    if mode == 'ellipse':
        outn = 8
    elif mode == 'circle':
        outn = 2
    else:
        raise(ValueError, 'argument mode invalid')
    return outn

##==========================CNR==========================#

def getCNRmodel(modelv, mode, device):
    outn = parsemode(mode)

    if modelv == 0:  # default
        CNRmodel = CNR(1, outn, nb_features=16)
    else:
        raise(ValueError, 'argument modelv invalid')
    return CNRmodel.to(device)



class CNR(nn.Module):
    def __init__(self, in_features, outn, nb_features=16):
        super(CNR, self).__init__()
        # 1 input image channel, 2 output channels, 32x32 input dimension
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(in_features, nb_features, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features, kernel_size=3,
                      stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features*2, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features * 2, nb_features * 2,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features * 2, nb_features * 4,
                      kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features * 4, nb_features * 4,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.FCBlock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, nb_features * 4),
            nn.ReLU(),
            nn.Linear(nb_features * 4, outn),
        )
        # todo double check features have been set correctly

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.FCBlock(x)
        return x


def getmodels(modelv, device):
    if modelv == 0:  # default
        refiner = Refiner(1, nb_features=64)
        discriminator = Discriminator(input_features=1)
    else:
        raise(ValueError, 'argument modelv invalid')
    return refiner.to(device), discriminator.to(device)

##==========================Refiner/Discriminator==========================#


class Refiner(nn.Module):
    def __init__(self, in_features, nb_features=64):
        super(Refiner, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_features, nb_features, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
        )

        self.ResnetBlock1 = nn.Sequential(
            nn.Conv2d(nb_features, nb_features, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        self.ResnetBlock2 = nn.Sequential(
            nn.Conv2d(nb_features, nb_features, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        self.ResnetBlock3 = nn.Sequential(
            nn.Conv2d(nb_features, nb_features, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        self.ResnetBlock4 = nn.Sequential(
            nn.Conv2d(nb_features, nb_features, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(nb_features, nb_features, 3, 1, 1, bias=True),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Conv2d(nb_features, in_features, 1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.ResnetBlock1(x)
        x = self.ResnetBlock2(x)
        x = self.ResnetBlock3(x)
        x = self.ResnetBlock4(x)
        output = self.out(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_features):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(input_features, 96, 3, 2, 1, bias=True),
            nn.ReLU(),

            nn.Conv2d(96, 64, 3, 2, 1, bias=True),
            nn.ReLU(),

            nn.MaxPool2d(3, 1, 0),

            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.ReLU(),

            nn.Conv2d(32, 2, 1, 1, 0, bias=True),
            nn.ReLU(),

            nn.Conv2d(2, 2, 1, 1, 0),
        )

    def forward(self, x):
        convs = self.convs(x)
        output = convs.view(convs.size(0), -1, 2)
        return output
