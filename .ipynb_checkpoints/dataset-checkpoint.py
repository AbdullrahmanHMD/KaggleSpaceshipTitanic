from torch.utils.data import Dataset
import torch

class PassengerDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels


    def __getitem__(self, index):
        datum = torch.Tensor(self.data[index])

        if self.labels is None:
            return datum

        return datum, self.labels[index]

    def __len__(self):
        return len(self.labels)