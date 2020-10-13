from .utils import *


filepath,_ = os.path.split(os.path.realpath(__file__))
filepath,_ = os.path.split(filepath)
filepath,_ = os.path.split(filepath)

# Set the appropriate paths of the datasets here.
_MINI_IMAGENET_DATASET_DIR = 'data/Mini-ImageNet/'


class MiniImageNet(ProtoData):
    img_size = (84, 84)
    def __init__(self, phase='train', augment='null', rot90_p=0., batch_size_down=8e4):

        self.base_folder = 'miniImagenet'
        self.img_size = (84, 84)
        assert(phase=='train' or phase == 'final' or phase=='val' or phase=='test')
        self.phase = phase
        self.name = 'MiniImageNet_' + phase

        print('Loading mini ImageNet dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_train_phase_train.pickle')
        file_val_categories_val_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_val.pickle')
        file_test_categories_test_phase = os.path.join(
            _MINI_IMAGENET_DATASET_DIR,
            'miniImageNet_category_split_test.pickle')

        if self.phase=='train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data = load_data(file_train_categories_train_phase)

        elif self.phase=='val':
            data = load_data(file_val_categories_val_phase)
        elif self.phase=='test':
            data = load_data(file_test_categories_test_phase)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        self.data = data['data']
        self.labels = data['labels']

        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)

        mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        self.augment = 'null' if self.phase == 'test' or self.phase == 'val' else augment
        if self.augment == 'null':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        elif self.augment == 'norm':
            self.transform = transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

    @property
    def dataset_dir(self):
        return _MINI_IMAGENET_DATASET_DIR

    def __repr__(self):
        string = self.__class__.__name__ + '(' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'augment=' + str(self.augment)
        if self.augment == 'w_rot90':
            string += ', ' + str(self.rot90)
        string += ')'
        return string

