import torch.nn as nn
import torch
import torch.nn.functional as F
from models.dropblock import DropBlock
import random
import numpy as np

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1, pool='max'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        if pool == 'max':
            self.pool = nn.MaxPool2d(stride)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0.
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        if self.training: self.num_batches_tracked += 1. / DropBlock.rate

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.pool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

    def train(self, mode=True):
        # if not mode and self.drop_block:
        #     self.num_batches_tracked += 2000.
        #     printl('train(self, mode='+str(mode)+'):' + str(self.num_batches_tracked))
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5, pool='max'):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate, pool=pool)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate, pool=pool)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, pool=pool)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size, pool=pool)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.block = block
        self.dropblock_size = dropblock_size
        self.pool = pool
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, pool='max'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, target= None, mixup_hidden=False, opt=None):
        if target is not None:
            if mixup_hidden:
                layer_mix = random.randint(0,3)
            else:
                layer_mix = None

            out = x

            target_a = target_b  = target

            out = self.layer1(out)

            if layer_mix == 0:
                out, target_a , target_b , lam = mixup_data(out, target, opt)

            out = self.layer2(out)

            if layer_mix == 1:
                out, target_a , target_b , lam  = mixup_data(out, target, opt)

            out = self.layer3(out)

            if layer_mix == 2:
                out, target_a , target_b , lam = mixup_data(out, target, opt)

            out = self.layer4(out)

            if  layer_mix == 3:
                out, target_a , target_b , lam = mixup_data(out, target, opt)

            if self.keep_avg_pool:
                out = self.avgpool(out)
            out = out.view(out.size(0), -1)

            return out , target_a , target_b, lam
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            if self.keep_avg_pool:
                x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return x

    def init_num_batches_tracked(self, num):
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.num_batches_tracked = num

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'block=' + str(self.block) + ', ' \
               + 'keep_prob=' + str(self.keep_prob) + ', ' \
               + 'avg_pool=' + str(self.keep_avg_pool) + ', ' \
               + 'drop_rate=' + str(self.drop_rate) + ', ' \
               + 'dropblock_size=' + str(self.dropblock_size) + ', ' \
               + 'pool=' + self.pool + ')'


def resnet12_mixup(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model



