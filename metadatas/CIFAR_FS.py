from .utils import *

filepath,_ = os.path.split(os.path.realpath(__file__))
filepath,_ = os.path.split(filepath)
filepath,_ = os.path.split(filepath)


# Set the appropriate paths of the datasets here.
_CIFAR_FS_DATASET_DIR = 'data/CIFAR-FS/'


class CIFAR_FS(ProtoData):
    def __init__(self, phase='train', augment='null', rot90_p=0., batch_size_down=8e4):

        assert (phase == 'train' or phase == 'final' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'CIFAR_FS_' + phase
        self.img_size = (32, 32)

        print('Loading CIFAR-FS dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_train.pickle')
        file_val_categories_val_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_val.pickle')
        file_test_categories_test_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_test.pickle')

        if self.phase == 'train':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data = load_data(file_train_categories_train_phase)
        elif self.phase == 'val':
            data = load_data(file_val_categories_val_phase)
        elif self.phase == 'test':
            data = load_data(file_test_categories_test_phase)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        self.data = data['data']
        self.labels = data['labels']

        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())

        self.num_cats = len(self.labelIds)

        mean_pix = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        self.augment = 'null' if self.phase == 'test' or self.phase == 'val' else augment
        if self.augment == 'null':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        elif self.augment == 'norm':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
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
        return _CIFAR_FS_DATASET_DIR

    def __repr__(self):
        string = self.__class__.__name__ + '(' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'augment=' + str(self.augment)
        if self.augment == 'w_rot90':
            string += ', ' + str(self.rot90)
        string += ')'
        return string
