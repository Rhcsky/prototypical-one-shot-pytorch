from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as dset
import shutil
import os
import torch
from glob import glob
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def get_dataloader(args, *modes):
    res = []
    for mode in modes:
        mdb_path = os.path.join('data', 'omniglot_' + mode + '.mdb')
        try:
            dataset = torch.load(mdb_path)
        except Exception:
            dataset = OmniglotDataset(mode)
            torch.save(dataset, mdb_path)

        if 'train' in mode:
            classes_per_it = args.classes_per_it_tr
            num_samples = args.num_support_tr + args.num_query_tr
        else:
            classes_per_it = args.classes_per_it_val
            num_samples = args.num_support_val + args.num_query_val

        sampler = PrototypicalBatchSampler(dataset.y, classes_per_it, num_samples, args.iterations)
        data_loader = DataLoader(dataset, batch_sampler=sampler)
        res.append(data_loader)

    if len(modes) == 1:
        return res[0]
    else:
        return res


class OmniglotDataset(Dataset):
    def __init__(self, mode='trainval'):
        super().__init__()
        self.root_dir = 'data/omniglot'
        self.vinyals_dir = 'data/vinyals'

        if not os.path.exists(self.root_dir):
            print('Data not found. Downloading data')
            self.download()

        self.x, self.y, self.class_to_idx = self.make_dataset(mode)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        origin_dir = 'data/omniglot-py'
        processed_dir = self.root_dir

        dset.Omniglot(root='data', background=False, download=True)
        dset.Omniglot(root='data', background=True, download=True)

        try:
            os.mkdir(processed_dir)
        except OSError:
            pass

        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(origin_dir, p)):
                shutil.move(os.path.join(origin_dir, p, f), processed_dir)

        shutil.rmtree(origin_dir)

    def make_dataset(self, mode):
        x = []
        y = []

        with open(os.path.join(self.vinyals_dir, mode + '.txt'), 'r') as f:
            classes = f.read().splitlines()

        class_to_idx = {string: i for i, string in enumerate(classes)}

        for idx, c in enumerate(tqdm(classes, desc="Making dataset")):
            class_dir, degree = c.rsplit('/', 1)
            degree = int(degree[3:])

            transform = A.Compose([
                A.Rotate((degree, degree), p=1),
                ToTensorV2(),
            ])

            for img_dir in glob(os.path.join(self.root_dir, class_dir, '*')):
                img = cv2.imread(img_dir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transform(image=img)['image']

                x.append(img)
                y.append(idx)

        return x, y, class_to_idx


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]  # 랜덤으로 클래스 60개 선택
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)  # 하나의 클래스당 선택한 이미지
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


if __name__ == '__main__':
    # torch.save(dataset, 'data/omniglot_trainval.mdb')
    dataset = torch.load('data/omniglot_trainval.mdb')
