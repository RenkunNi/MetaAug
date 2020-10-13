import torch.nn as nn
import torch
import random
import numpy as np
import pdb

# Embedding network used in Meta-learning with differentiable closed-form solvers
# (Bertinetto et al., in submission to NIPS 2018).
# They call the ridge rigressor version as "Ridge Regression Differentiable Discriminator (R2D2)."
  
# Note that they use a peculiar ordering of functions, namely conv-BN-pooling-lrelu,
# as opposed to the conventional one (conv-BN-lrelu-pooling).
  
def mixup_data(x, y, opt):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]//opt.episodes_per_batch

    xs, yas, ybs, ls = [], [], [], []

    for i in range(opt.episodes_per_batch):
        lam = np.random.beta(1., 1.)
        input_ = x[i * batch_size : (i+1) * batch_size]
        y_ = y[i]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * input_ + (1 - lam) * input_[index,:]
        y_a, y_b = y_, y_[index]
        xs.append(mixed_x)
        yas.append(y_a)
        ybs.append(y_b)
        ls.append(lam)

    return torch.cat(xs), torch.stack(yas), torch.stack(ybs), ls


def R2D2_conv_block(in_channels, out_channels, retain_activation=True, keep_prob=1.0):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2)
    )
    if retain_activation:
        block.add_module("LeakyReLU", nn.LeakyReLU(0.1))

    if keep_prob < 1.0:
        block.add_module("Dropout", nn.Dropout(p=1 - keep_prob, inplace=False))

    return block


class R2D2Embedding_mixup(nn.Module):
    def __init__(self, x_dim=3, h1_dim=96, h2_dim=192, h3_dim=384, z_dim=512, \
                 retain_last_activation=False):
        super(R2D2Embedding_mixup, self).__init__()

        self.block1 = R2D2_conv_block(x_dim, h1_dim)
        self.block2 = R2D2_conv_block(h1_dim, h2_dim)
        self.block3 = R2D2_conv_block(h2_dim, h3_dim, keep_prob=0.9)
        # In the last conv block, we disable activation function to boost the classification accuracy.
        # This trick was proposed by Gidaris et al. (CVPR 2018).
        # With this trick, the accuracy goes up from 50% to 51%.
        # Although the authors of R2D2 did not mention this trick in the paper,
        # we were unable to reproduce the result of Bertinetto et al. without resorting to this trick.
        self.block4 = R2D2_conv_block(h3_dim, z_dim, retain_activation=retain_last_activation, keep_prob=0.7)
  
    def forward(self, x, target=None, mixup_hidden=False, opt=None, layer_mix = None):
        if mixup_hidden:
            if not layer_mix:
                layer_mix = random.randint(0,4)

            if layer_mix == 0:
                x, y_a, y_b, lam = mixup_data(x, target, opt)

            b1 = self.block1(x)

            if layer_mix == 1:
                b1, y_a, y_b, lam = mixup_data(b1, target, opt)

            b2 = self.block2(b1)

            if layer_mix == 2:
                b2, y_a, y_b, lam = mixup_data(b2, target, opt)

            b3 = self.block3(b2)

            if layer_mix == 3:
                b3, y_a, y_b, lam = mixup_data(b3, target, opt)

            b4 = self.block4(b3)

            if layer_mix == 4:
                b4, y_a, y_b, lam = mixup_data(b4, target, opt)

            # Flatten and concatenate the output of the 3rd and 4th conv blocks as proposed in R2D2 paper.
            return torch.cat((b3.view(b3.size(0), -1), b4.view(b4.size(0), -1)), 1), y_a, y_b, lam

        else:
            b1 = self.block1(x)
            b2 = self.block2(b1)
            b3 = self.block3(b2)
            b4 = self.block4(b3)
            # Flatten and concatenate the output of the 3rd and 4th conv blocks as proposed in R2D2 paper.
            return torch.cat((b3.view(b3.size(0), -1), b4.view(b4.size(0), -1)), 1)
