import torch
import numpy
from torch.utils.data.dataset import Dataset
from data import Data

class EcgDataset(Dataset):
    def __init__(self, is_train = True):
        if is_train:
            self.data, self.label = Data().load_train_data("train_data")
        else:
            self.data, self.label = Data().load_test_data("test_data")

    def __getitem__(self, index):

        if isinstance(self.data, numpy.ndarray):
            self.data = torch.from_numpy(self.data)

        if isinstance(self.label, numpy.ndarray):
            self.label = torch.from_numpy(self.label)

        return self.data[index].float(), self.label[index].float()

    def __len__(self):

        return self.data.shape[0]
