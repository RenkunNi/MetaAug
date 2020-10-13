import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from kornia import augmentation as K
import kornia.augmentation.functional as KF
import kornia.augmentation.random_generator as Krg


def self_mix(data):
    size = data.size()
    W = size[-1]
    H = size[-2]
    ## uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    cut_w = W//2
    cut_h = H//2

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    while True:
        bbxn = np.random.randint(0, W-(bbx2-bbx1))
        bbyn = np.random.randint(0, H-(bby2-bby1))

        if bbxn != bbx1 or bbyn != bby1:
            break

    ## random rotate the croped when it's square
    if (bbx2 - bbx1) == (bby2 - bby1):
        k = random.sample([0, 1, 2, 3], 1)[0]
    else:
        k = 0
    data[:, :, bbx1:bbx2, bby1:bby2] = torch.rot90(data[:, :, bbxn:bbxn + (bbx2-bbx1), bbyn:bbyn + (bby2-bby1)], k, [2,3])

    return data


def rand_bbox(size, lam=0.5):
    W = size[-1]
    H = size[-2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    ## uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def build_grid(source_size,target_size):
    k = float(target_size)/float(source_size)
    direct = torch.linspace(-k,k,target_size).unsqueeze(0).repeat(target_size,1).unsqueeze(-1)
    full = torch.cat([direct,direct.transpose(1,0)],dim=2).unsqueeze(0)

    return full.cuda()


def random_crop_grid(x,grid):
    delta = x.size(-1)-grid.size(1)
    grid = grid.repeat(x.size(0),1,1,1).cuda()
    ## add random shifts by x
    grid[:,:,:,0] = grid[:,:,:,0]+ torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)
    ## add random shifts by y
    grid[:,:,:,1] = grid[:,:,:,1]+ torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)

    return grid


# random crop with gpu
def random_cropping(batch, t):
    batch = F.pad(batch.view([-1] + list(batch.shape[-3:])), (4,4,4,4))
    ## building central crop of t pixel size
    grid_source = build_grid(batch.size(-1),t)
    ## make radom shift for each batch
    grid_shifted = random_crop_grid(batch,grid_source)
    ## sample using grid sample
    sampled_batch = F.grid_sample(batch, grid_shifted, mode='nearest')

    return sampled_batch


def combine_labels(data, labels, train_way):
    for i, l in enumerate(range(train_way)):
        new_data = torch.cat((data[labels == l], data[labels == train_way + i]))
        new_order = torch.randperm(len(new_data))[:len(new_data)//2]
        data[labels == l] = new_data[new_order]

    return data[labels < train_way], labels[labels < train_way]


def large_rotation(data, opt):

    degrees = torch.randint(4,(data.shape[0] * data.shape[1],)) * 90

    out = KF.apply_rotation(
        data.view([-1] + list(data.shape[-3:])), {'degrees': torch.tensor(degrees), 'interpolation': torch.tensor([1]), 'align_corners': torch.tensor(True)})

    return out.view(data.shape)


def random_rotation(data, r, opt):

    num = data.shape[0] * data.shape[1]
    ppp = np.random.rand(num) < r
    degrees = torch.randint(-opt.rot_degree, opt.rot_degree, (num,)) * ppp
    out = KF.apply_rotation(
        data.view([-1] + list(data.shape[-3:])), {'degrees': torch.tensor(degrees), 'interpolation': torch.tensor([1]), 'align_corners': torch.tensor(True)})

    return out.view(data.shape)


def random_erase(data, opt):
    rec_er = K.RandomErasing(0.5, (.02, .4), (.3, 1/.3))
    out = rec_er(data.view([-1] + list(data.shape[-3:])))

    return out.view(data.shape)


def shearX(data, opt):
    ra = K.RandomAffine(degrees=0, shear=(-20,20))
    out = ra(data.view([-1] + list(data.shape[-3:])))

    return out.view(data.shape)


def drop_channel(data, opt):
    m = torch.nn.Dropout2d(p=0.5)
    out = m(data.view([-1] + list(data.shape[-3:])))

    return out.view(data.shape)


def shot_aug(data_support, labels_support, n_support, method, opt):
    size = data_support.shape
    if method == "fliplr":
        n_support = opt.s_du * n_support
        data_shot = flip(data_support, -1)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "flip_ver":
        n_support = opt.s_du * n_support
        data_shot = flip(data_support, -2)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "random_crop":
        n_support = opt.s_du * n_support
        data_shot = random_cropping(data_support, 32)
        data_shot2 = random_cropping(data_support, 32)
        data_support = torch.cat((data_shot.view([size[0], -1] + list(data_support.shape[-3:])), data_shot2.view([size[0], -1] + list(data_support.shape[-3:]))), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "random_rotation":
        n_support = opt.s_du * n_support
        data_shot = random_rotation(data_support, 1, opt)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "random_erase":
        n_support = opt.s_du * n_support
        data_shot = random_erase(data_support, opt)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "large_rotation":
        n_support = opt.s_du * n_support
        data_shot = large_rotation(data_support, labels_support, opt)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "random_solarize":
        n_support = opt.s_du * n_support
        data_shot = random_solarize(data_support, opt)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "shearX":
        n_support = opt.s_du * n_support
        data_shot = shearX(data_support, opt)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "self_aug":
        n_support = opt.s_du * n_support
        data_shot = torch.zeros_like(data_support)
        for ii in range(size[0]):
            data_shot[ii] = self_mix(data_support[ii])
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)

    return data_support, labels_support, n_support


def data_aug_mix(data, labels, r, method, opt):
    label_a, label_b = torch.zeros_like(labels), torch.zeros_like(labels)
    ls = []
    p = np.random.rand(1)

    if method == "mixup" and p < r:
        for ii in range(opt.episodes_per_batch):
            l = np.random.beta(1., 1.)
            data[ii], label_a[ii], label_b[ii] = mixup_data(data[ii], labels[ii], l, use_cuda=True)
            ls.append(l)
        data, label_a, label_b = map(Variable, (data, label_a, label_b))
    elif method == "cutmix" and p < r:
        for ii in range(opt.episodes_per_batch):
            lll = np.random.beta(2., 2.)
            rand_index = torch.randperm(data[ii].size()[0]).cuda()
            label_a[ii] = labels[ii]
            label_b[ii] = labels[ii][rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data[ii].size(), lll)
            data[ii][:, :, bbx1:bbx2, bby1:bby2] = data[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
            ## adjust lambda to exactly match pixel ratio
            lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data[ii].size()[-1] * data[ii].size()[-2]))
            ls.append(lll)
        data, label_a, label_b = map(Variable, (data, label_a, label_b))

    return data, label_a, label_b, ls, p


def data_aug(data, labels, r, method, opt):
    p = np.random.rand(1)

    if method == "combine" and p < r:
        data_ , label_ = [], []
        for ii in range(opt.episodes_per_batch):
            d, l = combine_labels(data[ii], labels[ii], opt.train_way)
            data_.append(d)
            label_.append(l)
        data = torch.stack(data_)
        labels = torch.stack(label_)
    elif method == "self_aug":
        for ii in range(opt.episodes_per_batch):
            data[ii] = self_mix(data[ii])
    elif method == "random_rotation":
        data = random_rotation(data, r, opt)
    elif method == "large_rotation" and p < r:
        data = large_rotation(data, labels, opt)
    elif method == "random_erase" and p < r:
        data = random_erase(data, opt)
    elif method == "random_solarize" and p < r:
        data = random_solarize(data, opt)
    elif method == "drop_channel" and p < r:
        data = drop_channel(data, opt)
    elif method == "shear" and p < r:
        data = shearX(data, opt)

    return data, labels

