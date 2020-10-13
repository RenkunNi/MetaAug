# Dataloader of Gidaris & Komodakis, CVPR 2018
# Adapted from:
# https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py
from __future__ import print_function

import os
import os.path
import numpy as np
import random
import pickle
import json
import math
import multiprocessing
from PIL import Image
from scipy.stats import beta

import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import Image
from PIL import ImageEnhance

import pdb


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


class ListDataset(object):
    """
    Args:
        elem_list (iterable/str): List of arguments which will be passed to
            `load` function. It can also be a path to file with each line
            containing the arguments to `load`
        load (function, optional): Function which loads the data.
            i-th sample is returned by `load(elem_list[i])`. By default `load`
            is identity i.e, `lambda x: x`
    """

    def __init__(self, elem_list, load=lambda x: x):
        self.list = elem_list
        self.load = load

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("CustomRange index out of range")
        return self.load(self.list[idx])


class FewShotDataloader(object):
    def __init__(self, dataset, kway=5, kshot=1, kquery=1, batch_size=1, num_workers=2,
                 epoch_size=2000, shuffle=True, fixed=False):

        self.dataset = dataset
        self.phase = self.dataset.phase

        max_possible_cate = self.dataset.num_cats
        assert(kway >= 0 and kway <= max_possible_cate)

        self.kway = kway
        self.kshot = kshot
        self.kquery = kquery

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size

        if fixed:
            self.file = os.path.join(self.dataset.dataset_dir, self.dataset.taskaug + '_' +
                                     self.phase + '_' + str(kway) + 'way_' + str(kshot) + 'shot_' + str(kquery) + 'query.pkl')
            self.tasks = pickle.load(open(self.file, 'rb'))\
                if os.path.exists(self.file)else self.create_alltasks()
            self.get_iterator = self.get_fixed_iterator
            shuffle = False

        self.shuffle = shuffle

    def sampleCategories(self, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from categories.

        Args:
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """

        assert(self.dataset.num_cats >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return self.dataset.sampleCategories(sample_size)

    def sample_examples_for_categories(self, categories, kshot, kquery):
        """
        Samples train and test examples of the categories.

        Args:
    	    categories: a list with the ids of the categories.
    	    kshot: the number of training examples per category that
                will be sampled.
            kquery: the number of test images that will be sampled
                from per the categories.

        Returns:
            shot_e: a list of length len(categories) * kshot with 2-element tuples.
                The 1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [0, dataset.num_cats - 1]).
            query_e: a list of length len(categories) * kquery with 2-element tuples.
                The 1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [0, dataset.num_cats - 1]).
        """


        if len(categories) == 0:
            return [], []

        shot_e = []
        query_e = []
        #for cate_idx in range(len(categories)):
        for cate_idx, cate in enumerate(categories):
            imd_ids = self.dataset.sampleImageIdsFrom(
                categories[cate_idx],
                sample_size=(kquery + kshot))

            imds_shot = imd_ids[kquery:]
            imds_query = imd_ids[:kquery]

            shot_e += [(img_id, cate_idx, cate) for img_id in imds_shot]
            query_e += [(img_id, cate_idx, cate) for img_id in imds_query]

        assert (len(shot_e) == len(categories) * kshot)
        assert(len(query_e) == len(categories) * kquery)

        random.shuffle(shot_e)
        random.shuffle(query_e)

        return shot_e, query_e

    '''
    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels
    '''

    def get_iterator(self, epoch=0):
        random.seed(epoch)
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        def load_function(iter_idx):
            categories = self.sampleCategories(self.kway)
            shot_e, query_e = self.sample_examples_for_categories(categories, self.kshot, self.kquery)
            Xt, Yt, Ot = self.dataset.createExamplesTensorData(query_e)
            if len(shot_e) > 0:
                Xe, Ye, Oe = self.dataset.createExamplesTensorData(shot_e)
                return Xe, Ye, Oe, Xt, Yt, Ot
            else:
                return Xt, Yt, Ot

        listdataset = ListDataset(elem_list=range(self.epoch_size), load=load_function)
        data_loader = DataLoader(listdataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

        return data_loader

    def create_alltasks(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        tasks = []
        for _ in range(self.epoch_size):
            categories = self.sampleCategories(self.kway)
            shot_e, query_e = self.sample_examples_for_categories(categories, self.kshot, self.kquery)
            tasks.append((shot_e, query_e))

        pickle.dump(tasks, open(self.file, 'wb'))
        return tasks

    def get_fixed_iterator(self, epoch=0):

        def load_function(iter_idx):
            shot_e, query_e = self.tasks[iter_idx]
            Xt, Yt, Ot = self.dataset.createExamplesTensorData(query_e)
            if len(shot_e) > 0:
                Xe, Ye, Oe = self.dataset.createExamplesTensorData(shot_e)
                return Xe, Ye, Oe, Xt, Yt, Ot
            else:
                return Xt, Yt, Ot

        listdataset = ListDataset(elem_list=range(self.epoch_size), load=load_function)
        data_loader = DataLoader(listdataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return int(self.epoch_size / self.batch_size)


#### For task level augment
class ProtoData(data.Dataset):
    taskaug = ''

    def sampleCategories(self, sample_size):
        return random.sample(self.labelIds, sample_size)

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.label2ind)
        assert(len(self.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.label2ind[cat_id], sample_size)

    def createExamplesTensorData(self, examples, method='support'):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
            dc_labels: a tensor of shape [nExamples] with the origial label (in 100)
                of each example.
        """
        
        images = torch.stack(
            [self[img_idx][0] for img_idx, _, _ in examples], dim=0)

        labels = torch.LongTensor([label for _, label, _ in examples])
        dc_labels = torch.LongTensor([label for _, _, label in examples])

        return images, labels, dc_labels


class Rotate90(object):
    def __init__(self, p, img_num_down=8e4):
        self.img_num = multiprocessing.Value("d", -1.)
        self.img_num_down = img_num_down
        self.p = 3. / 4. if p == -1 else p

    def __call__(self, img):
        self.img_num.value += 1.
        p = self.p * min(1., self.img_num.value / self.img_num_down)
        if random.random() < p:
            i = random.randint(0, 2)
            if i == 0:
                return img.transpose(Image.ROTATE_90)
            elif i == 1:
                return img.transpose(Image.ROTATE_180)
            elif i == 2:
                return img.transpose(Image.ROTATE_270)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'img_num_down=' + str(self.img_num_down) + ')'

