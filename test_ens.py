#-*- coding: utf-8 -*-
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm

from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12

from models.classification_heads import ClassificationHead, R2D2Head

from utils import pprint, set_gpu, Timer, count_accuracy, log

import numpy as np
import os
import pdb

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
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
    if opt.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()    
    elif opt.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif opt.head == 'R2D2':
        cls_head = R2D2Head().cuda()
    elif opt.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    else:
        print ("Cannot recognize the classification head type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_test, data_loader)

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
    delta = x.size(2)-grid.size(1)
    grid = grid.repeat(x.size(0),1,1,1).cuda()
    #Add random shifts by x
    grid[:,:,:,0] = grid[:,:,:,0]+ torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)
    #Add random shifts by y
    grid[:,:,:,1] = grid[:,:,:,1]+ torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)

    return grid

def random_cropping(batch, t):
    #Building central crop of t pixel size
    grid_source = build_grid(batch.size(-1),t)
    #Make radom shift for each batch
    grid_shifted = random_crop_grid(batch,grid_source)
    #Sample using grid sample
    sampled_batch = F.grid_sample(batch, grid_shifted, mode='nearest')

    return sampled_batch

def shot_aug(data_support, labels_support, n_support, method, opt):
    size = data_support.shape
    if method == "fliplr":
        n_support = opt.s_du * n_support
        data_shot = flip(data_support, -1)
        data_support = torch.cat((data_support, data_shot), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    elif method == "random_crop":
        n_support = opt.s_du * n_support
        data_shot = F.pad(data_support.view([-1] + list(data_support.shape[-3:])), (4,4,4,4))
        data_shot = random_cropping(data_shot, 32)
        data_support = torch.cat((data_support, data_shot.view([size[0], -1] + list(data_support.shape[-3:]))), dim = 1)
        labels_support = torch.cat((labels_support, labels_support), dim = 1)
    return data_support, labels_support, n_support


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./experiments/exp_1/best_model.pth',
                            help='path of the checkpoint file')
    parser.add_argument('--episode', type=int, default=1000,
                            help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                            help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--shot_aug', '-shotaug', default=[], nargs='+', type=str,
                            help='If use shot level data augmentation.')
    parser.add_argument('--s_du', type=int, default=1,
                            help='number of support examples augmented by shot')
    parser.add_argument('--query', type=int, default=15,
                            help='number of query examples per training class')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')

    opt = parser.parse_args()
    (dataset_test, data_loader) = get_dataset(opt)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    
    log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
    log(log_file_path, str(vars(opt)))

    # Define the models
    (embedding_net, cls_head) = get_model(opt)
    

    # Evaluate on test set
    test_accuracies = []
    for i, batch in enumerate(dloader_test(), 1):
        length = 0
        logits_sum = None
        for filename in os.listdir(opt.load):
            if filename.startswith("epoch"): 

                # Load saved model checkpoints
                saved_models = torch.load(opt.load + filename)
                embedding_net.load_state_dict(saved_models['embedding'])
                embedding_net.eval()
                cls_head.load_state_dict(saved_models['head'])
                cls_head.eval()

                with torch.no_grad():
                    data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
                    n_support = opt.way * opt.shot
                    n_query = opt.way * opt.query
                
                    for method in opt.shot_aug:
                        data_support, labels_support, n_support = shot_aug(data_support, labels_support, n_support, method, opt)

                    emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                    emb_support = emb_support.reshape(1, n_support, -1)
        
                    emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                    emb_query = emb_query.reshape(1, n_query, -1)

                if opt.head == 'SVM':
                    logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, maxIter=3)
                else:
                    logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)

                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits
 
                length += 1
        
        logits = logits_sum/length

        acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())
        
        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)
        
        if i % 50 == 0:
            print('Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                  .format(i, opt.episode, avg, ci95, acc))
