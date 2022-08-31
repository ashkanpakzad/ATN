# By Ashkan Pakzad (ashkanpakzad.github.io) 2022
# Adapted from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49

import torch
import torchvision
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(
        self, resize=True, feature_layers=[2], style_layers=[0, 1, 2, 3]
    ):
        """Set-up Feature and Style perceptual losses from ImageNet
        pretrained VGG-16. See Johnson et al. 16 for details.

        Adapted from
        https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49

        Parameters
        ----------
        resize : bool
            Resize to ImageNet input.
        feature_layers : list, int
            Which VGG-16 final block activation layers to use for computing
            feature loss.
        style_layers : list, int
            Which VGG-16 final block activation layers to use for computing
            style loss.
        """
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1).features[:4].eval())
        blocks.append(vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1).features[4:9].eval())
        blocks.append(vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1).features[9:16].eval())
        blocks.append(vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.feature_layers = feature_layers
        self.style_layers = style_layers

    def forward(self, yhat, yc, ys):
        """Calculate Feature and Style perceptual losses from ImageNet
        pretrained VGG-16.

        Parameters
        ----------
        yhat : float
            Output from image transformation network.
        yc : float
            Content target image.
        ys : float
            Style target image.

        Returns
        -------
        type float
            Sum of feature and style loss for input images.

        """
        if yhat.shape[1] != 3:
            yhat = yhat.repeat(1, 3, 1, 1)
            yc = yc.repeat(1, 3, 1, 1)
            ys = ys.repeat(1, 3, 1, 1)
        yhat = (yhat - self.mean) / self.std
        yc = (yc - self.mean) / self.std
        ys = (ys - self.mean) / self.std
        if self.resize:
            yhat = self.transform(
                yhat, mode="bilinear", size=(224, 224), align_corners=False
            )
            yc = self.transform(
                yc, mode="bilinear", size=(224, 224), align_corners=False
            )
            ys = self.transform(
                ys, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0

        for i, block in enumerate(self.blocks):
            yhat = block(yhat)
            yc = block(yc)
            ys = block(ys)
            if i in self.feature_layers:
                # default mean
                loss += torch.nn.functional.l1_loss(yhat, yc)
            if i in self.style_layers:
                act_x = yhat.reshape(yhat.shape[0], yhat.shape[1], -1)
                act_y = ys.reshape(ys.shape[0], ys.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                # default mean
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
