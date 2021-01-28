import os
import pickle
import shutil
import warnings
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as dset
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_dataloader(args, dataset, *modes):
    res = []
    print("Loading data...", end='')
    for mode in modes:
        if dataset == 'omniglot':
            mdb_path = os.path.join('data', 'omniglot_' + mode + '.mdb')
            try:
                dataset = torch.load(mdb_path)
            except Exception:
                dataset = OmniglotDataset(mode)
                torch.save(dataset, mdb_path)

        elif dataset == 'miniImagenet':
            mdb_path = os.path.join('data', 'miniImagenet_' + mode + '.mdb')
            try:
                dataset = torch.load(mdb_path)
            except Exception:
                dataset = MiniImagenetDataset(mode)
                torch.save(dataset, mdb_path)

        if 'train' in mode:
            classes_per_it = args.classes_per_it_tr
            num_support = args.num_support_tr
            num_query = args.num_query_tr
        else:
            classes_per_it = args.classes_per_it_val
            num_support = args.num_support_val
            num_query = args.num_query_val

        sampler = PrototypicalBatchSampler(dataset.y, classes_per_it, num_support, num_query, args.iterations)
        data_loader = DataLoader(dataset, batch_sampler=sampler)
        res.append(data_loader)

    print("done")
    if len(modes) == 1:
        return res[0]
    else:
        return res


class MiniImagenetDataset(Dataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.root_dir = 'data/miniImagenet'

        if not os.path.exists(self.root_dir):
            print('Data not found. Downloading data')
            self.download()

        dataset = pickle.load(open(os.path.join(self.root_dir, mode), 'rb'))

        self.x = dataset['image_data']

        self.y = torch.arange(len(self.x))
        for idx, (name, id) in enumerate(dataset['class_dict'].items()):
            s = slice(id[0], id[-1] + 1)
            self.y[s] = idx

    def __getitem__(self, index):

        img = self.x[index]

        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        x = transform(image=img)['image']

        return x, self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        import tarfile
        gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
        gz_filename = 'mini-imagenet.tar.gz'
        root = 'data/miniImagenet'

        self.download_file_from_google_drive(gdrive_id, root, gz_filename)

        filename = os.path.join(root, gz_filename)

        with tarfile.open(filename, 'r') as f:
            f.extractall(root)

        os.rename('data/miniImagenet/mini-imagenet-cache-train.pkl', 'data/miniImagenet/train')
        os.rename('data/miniImagenet/mini-imagenet-cache-val.pkl', 'data/miniImagenet/val')
        os.rename('data/miniImagenet/mini-imagenet-cache-test.pkl', 'data/miniImagenet/test')

    def download_file_from_google_drive(self, file_id, root, filename):
        from torchvision.datasets.utils import _get_confirm_token, _save_response_content

        """Download a Google Drive file from  and place it in root.
        Args:
            file_id (str): id of file to be downloaded
            root (str): Directory to place downloaded file in
            filename (str, optional): Name to save the file under. If None, use the id of the file.
            md5 (str, optional): MD5 checksum of the download. If None, do not check
        """
        # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
        import requests
        url = "https://docs.google.com/uc?export=download"

        root = os.path.expanduser(root)
        if not filename:
            filename = file_id
        fpath = os.path.join(root, filename)

        os.makedirs(root, exist_ok=True)

        if os.path.isfile(fpath):
            print('Using downloaded and verified file: ' + fpath)
        else:
            session = requests.Session()

            response = session.get(url, params={'id': file_id}, stream=True)
            token = _get_confirm_token(response)

            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(url, params=params, stream=True)

            _save_response_content(response, fpath)


class OmniglotDataset(Dataset):
    def __init__(self, mode='trainval'):
        super().__init__()
        self.root_dir = 'data/omniglot'
        self.vinyals_dir = 'data/vinyals/omniglot'

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
                A.Resize(28, 28),
                A.Rotate((degree, degree), p=1),
                A.Normalize(mean=0.92206, std=0.08426),
                ToTensorV2(),
            ])

            for img_dir in glob(os.path.join(self.root_dir, class_dir, '*')):
                img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transform(image=img)['image']

                x.append(img)
                y.append(idx)
        y = torch.LongTensor(y)
        return x, y, class_to_idx


class PrototypicalBatchSampler(object):
    """
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, labels, classes_per_it, num_samples_support, num_samples_query, iterations):
        """
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.num_samples_support = num_samples_support
        self.num_samples_query = num_samples_query
        self.iterations = iterations

        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
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
        """
        yield a batch of indexes
        """
        nss = self.num_samples_support
        nsq = self.num_samples_query
        cpi = self.classes_per_it

        for _ in range(self.iterations):
            batch_s = torch.LongTensor(nss * cpi)
            batch_q = torch.LongTensor(nsq * cpi)
            c_idxs = torch.randperm(len(self.classes))[:cpi]  # 랜덤으로 클래스 60개 선택
            for i, c in enumerate(self.classes[c_idxs]):
                s_s = slice(i * nss, (i + 1) * nss)  # 하나의 클래스당 선택한 support 이미지
                s_q = slice(i * nsq, (i + 1) * nsq)  # 하나의 클래스당 선택한 query 이미지

                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:nss + nsq]

                batch_s[s_s] = self.indexes[label_idx][sample_idxs][:nss]
                batch_q[s_q] = self.indexes[label_idx][sample_idxs][nss:]
            batch = torch.cat((batch_s, batch_q))
            yield batch

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        """
        return self.iterations
