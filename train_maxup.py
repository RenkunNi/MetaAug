# -*- coding: utf-8 -*-
import os
import argparse
import random
import numbers
import numpy as np
import torch
import math
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.autograd import Variable

from metadatas import *
from augmentations import *

from models.classification_heads import ClassificationHead, R2D2Head
from models.classification_heads import ClassificationHead_Mixup
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

from utils import set_gpu, Timer, count_accuracy, count_accuracy_mixup, check_dir, log

import pdb
import time

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'R2D2_mixup':
        network = R2D2Embedding_mixup().cuda()
    elif options.network == 'ResNet_mixup':
        network = resnet12_mixup(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network)
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
            network = torch.nn.DataParallel(network)
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = R2D2Head().cuda() 
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    if options.support_aug and 'mix' in options.support_aug:
        if options.head == 'R2D2':
            cls_head_mixup = R2D2Head_Mixup().cuda()
        elif options.head == 'SVM':
            cls_head_mixup = ClassificationHead_Mixup(base_learner='SVM-CS').cuda()
        else:
            print("Cannot recognize the dataset type")

        return (network, cls_head, cls_head_mixup)
        
    else:
        return (network, cls_head)


def get_datasets(name, phase, args):
    if name == 'miniImageNet':
        dataset = MiniImageNet(phase=phase, augment=args.feat_aug, rot90_p=args.t_p)  
    elif name == 'CIFAR_FS':
        dataset = CIFAR_FS(phase=phase, augment=args.feat_aug, rot90_p=args.t_p)
    elif name == 'FC100':
        dataset = FC100(phase=phase, augment=args.feat_aug, rot90_p=args.t_p)
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
    print(dataset)

    return dataset

def mixup_data(x, y, lam, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b

def mixup_criterion(opt, pred, y_a, y_b, lam):
    n_q = opt.train_way * opt.train_query
    b = opt.episodes_per_batch
    m = opt.m
    pred = pred.reshape(-1,opt.train_way)
    logit = F.log_softmax(pred, dim=-1)
    loss_a = F.nll_loss(logit, y_a.reshape(-1),reduction='none')
    loss_b = F.nll_loss(logit, y_b.reshape(-1),reduction='none')
    loss = loss_a.view(b,m,-1) * lam.view(b,m,1).cuda() + loss_b.view(b,m,-1) * (1 - lam).view(b,m,1).cuda()
    loss, loc = loss.max(dim=1)

    return loss.mean(), loc


def data_aug_(data_support, labels_support, data_query, labels_query, r, method, m, b, opt):
    b = opt.episodes_per_batch
    label_a, label_b = torch.zeros_like(labels_query), torch.zeros_like(labels_query)
    new_data_s = torch.zeros_like(data_support)
    new_data_q = torch.zeros_like(data_query)
    ls = []
    p = np.random.rand(1)

    if method == "qcm":
        for ii in range(b):
            lll = np.random.beta(2., 2.)
            rand_index = torch.randperm(data_query[ii].size()[0]).cuda()
            label_a[ii] = labels_query[ii]
            label_b[ii] = labels_query[ii][rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data_query[ii].size(), lll)
            new_data_q[ii] = data_query[ii]
            new_data_q[ii][:, :, bbx1:bbx2, bby1:bby2] = data_query[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data_query[ii].size()[-1] * data_query[ii].size()[-2]))  
            ls.append(lll)
            new_data_s = data_support
    elif method == "qre":
        new_data_q = random_erase(data_query, opt)
        new_data_s = data_support
        label_a, label_b = labels_query, labels_query
        ls = [1.] * b
    elif method == "sre":
        new_data_s = random_erase(data_support, opt)
        new_data_q = data_query
        label_a, label_b = labels_query, labels_query
        ls = [1.] * b
    elif method == "tlr":
        for ii in range(b):
            for j in range(opt.train_way):
                k = random.sample([0, 0, 0, 1, 2, 3], 1)[0]
                new_data_s[ii][labels_support[ii] == j] = torch.rot90(data_support[ii][labels_support[ii] == j], k, [2, 3])
                new_data_q[ii][labels_query[ii] == j] = torch.rot90(data_query[ii][labels_query[ii] == j], k, [2, 3])
        label_a, label_b = labels_query, labels_query
        ls = [1.] * b
    elif method == "qsm":
        for ii in range(b):
            new_data_q[ii] = self_mix(data_query[ii])
        new_data_s = data_support
        label_a, label_b = labels_query, labels_query
        ls = [1.] * b
    elif method == "ssm":
        for ii in range(b):
            new_data_s[ii] = self_mix(data_support[ii])
        new_data_q = data_query
        label_a, label_b = labels_query, labels_query
        ls = [1.] * b
    elif method == "qcm + tlr":
        for ii in range(b):
            for j in range(opt.train_way):
                k = random.sample([0, 0, 0, 1, 2, 3], 1)[0]
                new_data_s[ii][labels_support[ii] == j] = torch.rot90(data_support[ii][labels_support[ii] == j], k, [2, 3])
                new_data_q[ii][labels_query[ii] == j] = torch.rot90(data_query[ii][labels_query[ii] == j], k, [2, 3])

            lll = np.random.beta(2., 2.)
            rand_index = torch.randperm(new_data_q[ii].size()[0]).cuda()
            label_a[ii] = labels_query[ii]
            label_b[ii] = labels_query[ii][rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(new_data_q[ii].size(), lll)
            new_data_q[ii][:, :, bbx1:bbx2, bby1:bby2] = new_data_q[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (new_data_q[ii].size()[-1] * new_data_q[ii].size()[-2]))
            ls.append(lll)
    elif method == "qsm + tlr":
        for ii in range(b):
            for j in range(opt.train_way):
                k = random.sample([0, 0, 0, 1, 2, 3], 1)[0]
                new_data_s[ii][labels_support[ii] == j] = torch.rot90(data_support[ii][labels_support[ii] == j], k, [2, 3]) 
                new_data_q[ii][labels_query[ii] == j] = torch.rot90(data_query[ii][labels_query[ii] == j], k, [2, 3])   

            new_data_q[ii] = self_mix(data_query[ii])
        label_a, label_b = labels_query, labels_query
        ls = [1.] * b
    elif method == "qre + tlr":
        for ii in range(b):
            for j in range(opt.train_way):
                k = random.sample([0, 0, 0, 1, 2, 3], 1)[0]
                new_data_s[ii][labels_support[ii] == j] = torch.rot90(data_support[ii][labels_support[ii] == j], k, [2, 3]) 
                new_data_q[ii][labels_query[ii] == j] = torch.rot90(data_query[ii][labels_query[ii] == j], k, [2, 3])   

        new_data_q = random_erase(new_data_q, opt)
        label_a, label_b = labels_query, labels_query
        ls = [1.] * b
    elif method == "qcm + qre":
        new_data_q = random_erase(data_query, opt)
        new_data_s = data_support
        for ii in range(b):
            lll = np.random.beta(2., 2.)
            rand_index = torch.randperm(new_data_q[ii].size()[0]).cuda()
            label_a[ii] = labels_query[ii]
            label_b[ii] = labels_query[ii][rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(new_data_q[ii].size(), lll)
            new_data_q[ii][:, :, bbx1:bbx2, bby1:bby2] = new_data_q[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (new_data_q[ii].size()[-1] * new_data_q[ii].size()[-2]))
            ls.append(lll)
    elif method == "qcm + qsm":
        new_data_s = data_support
        for ii in range(b):
            new_data_q[ii] = self_mix(data_query[ii])
            lll = np.random.beta(2., 2.)
            rand_index = torch.randperm(new_data_q[ii].size()[0]).cuda()
            label_a[ii] = labels_query[ii]
            label_b[ii] = labels_query[ii][rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(new_data_q[ii].size(), lll)
            new_data_q[ii][:, :, bbx1:bbx2, bby1:bby2] = new_data_q[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (new_data_q[ii].size()[-1] * new_data_q[ii].size()[-2]))
            ls.append(lll)
    elif method == "qcm + ssm":
        for ii in range(b):
            new_data_s[ii] = self_mix(data_support[ii])
            lll = np.random.beta(2., 2.)
            rand_index = torch.randperm(data_query[ii].size()[0]).cuda()
            label_a[ii] = labels_query[ii]
            label_b[ii] = labels_query[ii][rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data_query[ii].size(), lll)
            new_data_q[ii][:, :, bbx1:bbx2, bby1:bby2] = data_query[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (new_data_q[ii].size()[-1] * new_data_q[ii].size()[-2]))
            ls.append(lll)
    elif method == "qcm + sre":
        for ii in range(b):
            lll = np.random.beta(2., 2.)
            rand_index = torch.randperm(data_query[ii].size()[0]).cuda()
            label_a[ii] = labels_query[ii]
            label_b[ii] = labels_query[ii][rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(data_query[ii].size(), lll)
            new_data_q[ii] = data_query[ii]
            new_data_q[ii][:, :, bbx1:bbx2, bby1:bby2] = data_query[ii][rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lll = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data_query[ii].size()[-1] * data_query[ii].size()[-2]))
            ls.append(lll)
        new_data_s = random_erase(data_support)
    else:
        new_data_s = data_support
        new_data_q = data_query
        label_a, label_b = labels_query, labels_query
        ls = [1.] * b

    return new_data_s, labels_support, new_data_q, label_a, label_b, torch.tensor(ls)


def data_aug_maxup(data_support, labels_support, data_query, labels_query, r, methods, opt):
    train_n_query = opt.train_way * opt.train_query
    train_n_support = opt.train_way * opt.train_shot * opt.s_du
    b = opt.episodes_per_batch
    n_max = opt.m
    label_a = torch.zeros(n_max, b, train_n_query).long()
    label_b = torch.zeros(n_max, b, train_n_query).long()
    label_s = torch.zeros(n_max, b, train_n_support).long()
    new_data_q = torch.zeros([n_max, b, train_n_query] + list(data_query.shape[-3:]))
    new_data_s = torch.zeros([n_max, b, train_n_support] + list(data_support.shape[-3:]))
    ls = torch.zeros(n_max, b)

    for m in range(n_max):
        method = random.sample(methods, 1)[0]
        new_data_s[m], label_s[m], new_data_q[m], label_a[m], label_b[m], ls[m] = data_aug_(data_support, labels_support, data_query, labels_query, r, method, m, b, opt)

    new_data_s.transpose_(0,1)
    new_data_q.transpose_(0,1)
    label_s.transpose_(0,1)
    label_a.transpose_(0,1)
    label_b.transpose_(0,1)
    ls.transpose_(0,1)

    return new_data_s, label_s, new_data_q, label_a, label_b, ls.reshape(-1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=5,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--sample-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--num-per-batch', type=int, default=1000,
                            help='number of episodes per batch')
    parser.add_argument('--m', type=int, default=4,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')

    parser.add_argument('--load', default=None,
                            help='path of the checkpoint file')
    parser.add_argument('--pretrain', type=str, default=None,
                            help='path of the checkpoint file')

    ## Data Augmentation
    parser.add_argument('--feat_aug', '-faug', default='norm', type=str,
                        help='If use feature level augmentation.')
    parser.add_argument('--task_aug', '-taug', default=[], nargs='+', type=str,
                        help='If use task level data augmentation.')
    parser.add_argument('--support_aug', '-saug', default=None, type=str,
                        help='If use support level data augmentation.')
    parser.add_argument('--shot_aug', '-shotaug', default=[], nargs='+', type=str,
                        help='If use shot level data augmentation.')
    parser.add_argument('--query_aug', '-qaug', default=None, type=str,
                        help='If use query level data augmentation.')
    parser.add_argument('--t_p', '-tp', default=1, type=float,
                        help='The possibility of sampling categories or images with rot90.')
    parser.add_argument('--s_p', '-sp', default=1, type=float,
                        help='The possibility of sampling categories or images with rot90.')
    parser.add_argument('--s_du', '-sdu', default=1, type=int,
                        help='The possibility of sampling categories or images with rot90.')
    parser.add_argument('--q_p', '-qp', default=1, type=float,
                        help='The possibility of sampling categories or images with rot90.')
    parser.add_argument('--rot_degree', default=30, type=int,
                        help='Degree for random rotation.')

    opt = parser.parse_args()
    
    trainset = get_datasets(opt.dataset, 'train', opt)
    valset = get_datasets(opt.dataset, 'val', opt)
  
    epoch_size = opt.episodes_per_batch * opt.num_per_batch

    dloader_train = FewShotDataloader(trainset, kway=opt.sample_way, kshot=opt.train_shot, kquery=opt.train_query,
                                    batch_size=opt.episodes_per_batch, num_workers=4, epoch_size=epoch_size, shuffle=True)
    dloader_val = FewShotDataloader(valset, kway=opt.train_way, kshot=opt.val_shot, kquery=opt.val_query,
                                  batch_size=1, num_workers=1, epoch_size=2000, shuffle=False, fixed=False)

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    if opt.support_aug and "mix" in opt.support_aug:
        (embedding_net, cls_head, cls_head_mixup) = get_model(opt)
        embedding_net.cuda()
        cls_head.cuda()
        cls_head_mixup.cuda()
    else:
        (embedding_net, cls_head) = get_model(opt)
        embedding_net.cuda()
        cls_head.cuda()

    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                 {'params': cls_head.parameters()}], lr=0.1, momentum=0.9, \
                                          weight_decay=5e-4, nesterov=True)
    
    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024 if e < 60 else (0.001)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    da_pool = ['qcm', 'qre', 'sre', 'tlr', 'qcm + tlr', 'qre + tlr']
    #da_pool = ['qcm', 'qre', 'sre', 'tlr', 'qcm + tlr', 'qre + tlr', 'qre + sre', 'qcm + qre', 'qcm + sre']
    
    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_s_accuracies = []
        train_losses = []

        for i, batch in enumerate(dloader_train(epoch), 1):
            if "random_crop" in opt.shot_aug:
                data_support, labels_support, _, data_query, labels_query, _ = [x.cuda() for x in batch]
            else:
                data_support, labels_support, _, data_query, labels_query, _ = [x for x in batch]

            train_n_support = opt.train_way * opt.train_shot 
            train_n_query = opt.train_way * opt.train_query 
            rs, rq = 0., 0.

            ## data augmentation for shots (increasing num of shots for support)
            for shot_method in opt.shot_aug:
                data_support, labels_support, train_n_support = shot_aug(data_support, labels_support, train_n_support, shot_method, opt)

            new_data_s, label_s, new_data_q, label_a, label_b, ls = data_aug_maxup(data_support, labels_support, data_query, labels_query, opt.q_p, da_pool, opt)

            new_data_s, new_data_q, label_a, label_b, ls, labels_support = new_data_s.cuda(), new_data_q.cuda(), label_a.cuda(), label_b.cuda(), ls.cuda(), labels_support.cuda()
            label_s = label_s.cuda()


            ## get embedding
            emb_support = embedding_net(new_data_s.reshape([-1] + list(new_data_s.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch*opt.m, train_n_support, -1)
            
            emb_query = embedding_net(new_data_q.reshape([-1] + list(new_data_q.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch*opt.m, train_n_query, -1)
            
            ## get logits for query embedding
            logit_query = cls_head(emb_query, emb_support, label_s, opt.train_way, opt.train_shot)

            ## get loss for the outer loop
            loss, loc = mixup_criterion(opt, logit_query.reshape(opt.episodes_per_batch, opt.m*train_n_query, -1), label_a, label_b, ls)

            ## pick the worst case
            acc = count_accuracy_mixup(logit_query, label_a.reshape(opt.episodes_per_batch * opt.m, -1), label_b.reshape(opt.episodes_per_batch * opt.m, -1), ls)
            
            ## get accuracies
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                train_acc_avg_s = np.mean(np.array(train_s_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []
        
        for i, batch in enumerate(dloader_val(epoch), 1):
            data_support, labels_support, _, data_query, labels_query, _ = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query
 
            for method in opt.shot_aug:
                data_support, labels_support, test_n_support = shot_aug(data_support, labels_support, test_n_support, method, opt)

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)[0]

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())
            
        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci = np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        lr_scheduler.step()

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
