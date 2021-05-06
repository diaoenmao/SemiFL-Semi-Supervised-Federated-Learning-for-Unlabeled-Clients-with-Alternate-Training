import anytree
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class STL10(Dataset):
    data_name = 'STL10'
    file = [('http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz', '91f7769df0f17e558f3565bffb0c7dfb')]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                          mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.other = {'id': id}

    def __getitem__(self, index):
        data, target = Image.fromarray(self.data[index]), torch.tensor(self.target[index])
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**other, 'data': data, 'target': target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_labeled_data, train_labeled_target = read_data_file(
            os.path.join(self.raw_folder, 'stl10_binary', 'train_X.bin'),
            os.path.join(self.raw_folder, 'stl10_binary', 'train_y.bin'))
        train_unlabeled_data, train_unlabeled_target = read_data_file(
            os.path.join(self.raw_folder, 'stl10_binary', 'unlabeled_X.bin'))
        train_data = np.concatenate((train_labeled_data, train_unlabeled_data))
        train_target = np.concatenate((train_labeled_target, train_unlabeled_target))
        test_data, test_target = read_data_file(
            os.path.join(self.raw_folder, 'stl10_binary', 'test_X.bin'),
            os.path.join(self.raw_folder, 'stl10_binary', 'test_y.bin'))
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        classes = list(map(str, list(range(10))))
        for c in classes:
            make_tree(classes_to_labels, [c])
        classes_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (
            classes_to_labels, classes_size)


def read_data_file(data_path, label_path=None):
    with open(data_path, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
    if label_path is not None:
        with open(label_path, 'rb') as f:
            labels = (np.fromfile(f, dtype=np.uint8) - 1).astype(np.int64)  # 0-based
    else:
        labels = - np.ones(images.shape[0], dtype=np.int64)
    return images, labels
