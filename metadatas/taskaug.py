from __future__ import print_function

import numpy as np
import random
import math
import multiprocessing
import pdb
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TranF

from .utils import ProtoData


class DualCategories(data.Dataset):
    def __init__(self, dataset, p=0.5, std=0.1, batch_size_down=4e4):
        self.dataset = dataset
        self.std = std
        self.batch_num = multiprocessing.Value("d", -1.)
        self.batch_size_down = batch_size_down

        self.phase = self.dataset.phase
        self.num_cats_new = self.dataset.num_cats * (self.dataset.num_cats - 1) // 2
        self.num_cats = self.dataset.num_cats + self.num_cats_new

        if p == -1:
            self.p = float(self.num_cats_new) / self.num_cats
        else:
            self.p = p

    def sampleCategories(self, sample_size):
        self.batch_num.value += 1
        p = self.p * (self.batch_size_down - self.batch_num.value) / self.batch_size_down
        sample1_size = np.sum(np.random.rand(sample_size) > p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.num_cats_new, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

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
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        if cat_id < self.dataset.num_cats:
            return [(False, d_id) for d_id in self.dataset.sampleImageIdsFrom(
                self.dataset.labelIds[cat_id], sample_size)]
        else:
            for cat_id in range(self.dataset.num_cats, self.dataset.num_cats + self.num_cats_new):
                cat_id = cat_id - self.dataset.num_cats
                cat1_id = int((-1 + math.sqrt(1 + 8 * cat_id)) / 2)
                cat2_id = int(cat_id - cat1_id * (cat1_id + 1) / 2)
                cat1_id += 1
            ids1 = self.dataset.sampleImageIdsFrom(self.dataset.labelIds[cat1_id], sample_size)
            ids2 = self.dataset.sampleImageIdsFrom(self.dataset.labelIds[cat2_id], sample_size)
            return [(True, d_id) for d_id in zip(ids1, ids2)]

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

        dataset = self.dataset
        def get_image(img_idx):
            if img_idx[0]:
                img_idx = img_idx[1]
                image1 = dataset[img_idx[0]][0]
                image2 = dataset[img_idx[1]][0]
                shift = np.random.normal(loc=0, scale=self.std)
                return image1 * (shift / 2.) + image2 * ((1 - shift) / 2.)
            else:
                return dataset[img_idx[1]][0]

        images = torch.stack(
            [get_image(img_idx) for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'std=' + str(self.std) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats_new=' + str(self.num_cats_new) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ', ' \
               + 'batch_size_down=' + str(self.batch_size_down) + ')'


class PermuteChannels(data.Dataset):
    def __init__(self, dataset, p=-1, ):
        self.dataset = dataset

        self.phase = self.dataset.phase
        self.num_cats_new = self.dataset.num_cats * 5
        self.num_cats = self.dataset.num_cats + self.num_cats_new

        self.orders = (torch.LongTensor([0, 1, 2]), torch.LongTensor([1, 2, 0]),
                       torch.LongTensor([2, 0, 1]), torch.LongTensor([1, 0, 2]),
                       torch.LongTensor([0, 2, 1]), torch.LongTensor([2, 1, 0]))

        if p == -1:
            self.p = 5./6.
        else:
            self.p = p

    def sampleCategories(self, sample_size):
        sample1_size = np.sum(np.random.rand(sample_size) > self.p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.num_cats_new, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

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
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        perm_id = cat_id // self.dataset.num_cats
        cat_id = cat_id % self.dataset.num_cats
        return [(perm_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

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

        dataset = self.dataset

        images = torch.stack(
            [dataset[img_idx[1]][0][self.orders[img_idx[0]]] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats_new=' + str(self.num_cats_new) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ')'


class DropChannels(data.Dataset):
    def __init__(self, dataset, p=-1):
        self.dataset = dataset

        self.phase = self.dataset.phase
        self.num_cats_new = self.dataset.num_cats * 6
        self.num_cats = self.dataset.num_cats + self.num_cats_new

        self.orders = (torch.LongTensor([1, 1, 1]), torch.LongTensor([0, 1, 0]),
                       torch.LongTensor([1, 0, 0]), torch.LongTensor([1, 1, 0]),
                       torch.LongTensor([1, 0, 1]), torch.LongTensor([0, 1, 1]),
                       torch.LongTensor([0, 0, 1]))

        if p == -1:
            self.p = 5./6.
        else:
            self.p = p

    def sampleCategories(self, sample_size):
        sample1_size = np.sum(np.random.rand(sample_size) > self.p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.num_cats_new, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

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
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        ch_id = cat_id // self.dataset.num_cats
        cat_id = cat_id % self.dataset.num_cats
        return [(ch_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

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

        dataset = self.dataset
        

        images = torch.stack(
            [dataset[img_idx[1]][0] * self.orders[img_idx[0]].view(3,1,1) for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats_new=' + str(self.num_cats_new) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ')'


class Rot90(data.Dataset):
    def __init__(self, dataset, p=-1, batch_size_down=8e4):
        self.dataset = dataset
        self.batch_num = multiprocessing.Value("d", -1.)
        self.batch_size_down = batch_size_down

        self.phase = self.dataset.phase
        self.num_cats_new = self.dataset.num_cats * 3
        self.num_cats = self.dataset.num_cats + self.num_cats_new

        if p == -1:
            self.p = float(self.num_cats_new) / self.num_cats
        else:
            self.p = p

    def sampleCategories(self, sample_size):
        self.batch_num.value += 1.
        sample1_size = np.sum(np.random.rand(sample_size) > self.p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.num_cats_new, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

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
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        rot90_id = int(cat_id // self.dataset.num_cats)
        cat_id = cat_id % self.dataset.num_cats
        return [(rot90_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

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

        dataset = self.dataset

        images = torch.stack(
            [torch.rot90(dataset[img_idx[1]][0], img_idx[0], [1, 2]) for img_idx, _, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label, _ in examples])
        dc_labels = torch.LongTensor([label for _, _, label in examples])
        return images, labels, dc_labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats_new=' + str(self.num_cats_new) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ', ' \
               + 'batch_size_down=' + str(self.batch_size_down) + ')'


class AddNoise(data.Dataset):
    def __init__(self, dataset, p=-1, batch_size_down=8e4):
        self.dataset = dataset
        self.batch_num = multiprocessing.Value("d", -1.)
        self.batch_size_down = batch_size_down

        self.phase = self.dataset.phase
        self.num_cats = self.dataset.num_cats

        if p == -1:
            self.p = float(self.num_cats_new) / self.num_cats
        else:
            self.p = p

    def sampleCategories(self, sample_size):
        self.batch_num.value += 1.
        sample1_size = np.sum(np.random.rand(sample_size) > self.p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.dataset.num_cats, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

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
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        delta = random.uniform(0, 0.5)
        add_id = int(cat_id // self.dataset.num_cats)
        cat_id = cat_id % self.dataset.num_cats
        return [(add_id * delta, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

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

        dataset = self.dataset
        
        images = torch.stack(
            [dataset[img_idx[1]][0] + torch.randn_like(dataset[img_idx[1]][0]) * np.sqrt(img_idx[0]) for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ', ' \
               + 'batch_size_down=' + str(self.batch_size_down) + ')'

class TaskAug(data.Dataset):
    def __init__(self, dataset, method, p=-1, batch_size_down=8e4):
        self.dataset = dataset
        self.batch_num = multiprocessing.Value("d", -1.)
        self.batch_size_down = batch_size_down

        self.phase = self.dataset.phase
        self.num_cats = self.dataset.num_cats 
        self.method = method

        self.test = np.random.randint(50)

        if p == -1:
            self.p = float(self.num_cats_new) / self.num_cats
        else:
            self.p = p

    @staticmethod
    def rand_bbox(size, lam=0.5):
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


    def sampleCategories(self, sample_size):
        self.batch_num.value += 1.
        p = self.p #* min(1., self.batch_num.value / self.batch_size_down)
        sample1_size = np.sum(np.random.rand(sample_size) > p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample_size, replace=False)
        mix_sample = np.random.choice(sample_size, sample2_size, replace=False)
        for ms in mix_sample:
            sample1[ms] += self.dataset.num_cats
        sample2 = np.random.choice(self.dataset.num_cats, sample2_size, replace=False) + self.dataset.num_cats * 2
        
        return list(sample1) + list(sample2)

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
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        group_id = int(cat_id // self.dataset.num_cats)
        cat_id = cat_id % self.dataset.num_cats

        if self.method == "Mix" or self.method == "Combine":
            return [(group_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
                self.dataset.labelIds[cat_id], sample_size)]
        elif self.method == "CutMix":
            lam = np.random.beta(2., 2.)
            rbbx = self.rand_bbox(self.dataset.img_size, lam)
            return [(group_id, d_id, rbbx) for d_id in self.dataset.sampleImageIdsFrom(
                self.dataset.labelIds[cat_id], sample_size)]

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

        dataset = self.dataset

        images = torch.stack([dataset[img_idx[1]][0] for img_idx, _ in examples if img_idx[0] <= 1], dim=0)
        labels = torch.LongTensor([label for img_idx, label in examples if img_idx[0] <= 1])

        labels_2 = torch.LongTensor([label for img_idx, label in examples if img_idx[0] == 1])
        labels_3 = torch.LongTensor([label for img_idx, label in examples if img_idx[0] == 2])
        assert(len(labels_2) == len(labels_3))

        if len(labels_2) > 0:
            images_2 = torch.stack([dataset[img_idx[1]][0] for img_idx, _ in examples if img_idx[0] == 1], dim=0)
            images_3 = torch.stack([dataset[img_idx[1]][0] for img_idx, _ in examples if img_idx[0] == 2], dim=0)
       
            uni_l2 = torch.unique(labels_2)
            n2 = len(uni_l2)
            uni_l = torch.unique(labels)
            if self.method == "Mix":
                for i, l2 in enumerate(uni_l2):
                    images[labels == l2] = (images_2[labels_2 == l2] + images_3[labels_3 == len(uni_l) + i])/2
            elif self.method == "CutMix":
                labels_all = torch.LongTensor([label for img_idx, label in examples])
                for i, l2 in enumerate(uni_l2):
                    lam = np.random.beta(2., 2.)
                    for j in range(len(images[labels == l2])):
                        bbx1, bby1, bbx2, bby2 = self.rand_bbox(self.dataset.img_size, lam)
                        images[labels == l2][j][:, bbx1:bbx2, bby1:bby2] = images_3[labels_3 == len(uni_l) + i][j][:, bbx1:bbx2, bby1:bby2]
            elif self.method == "Combine":
                for i, l2 in enumerate(uni_l2):
                    new_images = torch.cat((images_2[labels_2 == l2], images_3[labels_3 == len(uni_l) + i]))
                    new_order = torch.randperm(len(new_images))[:len(new_images)//2]
                    images[labels == l2] = new_images[new_order]

        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ', ' \
               + 'batch_size_down=' + str(self.batch_size_down) + ')'


class RE(data.Dataset):
    def __init__(self, dataset, p=1):
        self.dataset = dataset

        self.phase = self.dataset.phase
        self.num_cats = self.dataset.num_cats

        self.p = p

    @staticmethod
    def rand_bbox(size, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):

        for attempt in range(100):
            area = size[0] * size[1]
       
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < size[0] and h < size[1]:
                x1 = random.randint(0, size[1] - h)
                y1 = random.randint(0, size[0] - w)
                x2 = x1 + h
                y2 = y1 + w
            
                return x1, y1, x2, y2, mean


    def sampleCategories(self, sample_size):
        sample1_size = np.sum(np.random.rand(sample_size) > self.p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.dataset.num_cats, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

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
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        group_id = int(cat_id // self.dataset.num_cats)
        cat_id = cat_id % self.dataset.num_cats

        #rbbx = self.rand_bbox(self.dataset.img_size, lam)
        return [(group_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

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

        dataset = self.dataset

        images = torch.stack([dataset[img_idx[1]][0] for img_idx, _ in examples if img_idx[0] <= 1], dim=0)
        labels = torch.LongTensor([label for img_idx, label in examples if img_idx[0] <= 1])

        labels_2 = torch.LongTensor([label for img_idx, label in examples if img_idx[0] == 1])

        if len(labels_2) > 0:
            uni_l2 = torch.unique(labels_2)
            for i, l2 in enumerate(uni_l2):
                for j in range(len(images[labels == l2])):
                    bbx1, bby1, bbx2, bby2, mean = self.rand_bbox(self.dataset.img_size)
                    images[labels == l2][j][0, bbx1:bbx2, bby1:bby2] = mean[0]
                    images[labels == l2][j][1, bbx1:bbx2, bby1:bby2] = mean[1]
                    images[labels == l2][j][2, bbx1:bbx2, bby1:bby2] = mean[2]

        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ')'


class Solarize(data.Dataset):
    def __init__(self, dataset, p=1):
        self.dataset = dataset

        self.phase = self.dataset.phase
        self.num_cats = self.dataset.num_cats

        self.p = p

    @staticmethod
    def random_solarize(data):
        solarize = K.RandomSolarize(0.1, 0.1, same_on_batch=True)
        out = solarize(data.view([-1] + list(data.shape[-3:])))

        return out.view(data.shape)


    def sampleCategories(self, sample_size):
        sample1_size = np.sum(np.random.rand(sample_size) > self.p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.dataset.num_cats, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

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
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        group_id = int(cat_id // self.dataset.num_cats)
        cat_id = cat_id % self.dataset.num_cats

        return [(group_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

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

        dataset = self.dataset

        images = torch.stack([dataset[img_idx[1]][0] for img_idx, _ in examples if img_idx[0] <= 1], dim=0)
        labels = torch.LongTensor([label for img_idx, label in examples if img_idx[0] <= 1])

        labels_2 = torch.LongTensor([label for img_idx, label in examples if img_idx[0] == 1])

        if len(labels_2) > 0:
            uni_l2 = torch.unique(labels_2)
            for i, l2 in enumerate(uni_l2):
                images[labels == l2] = self.random_solarize(images[labels == l2])

        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ')'

