import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio


class CustomDataset_txt(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        self.train_data = data_all[data['train_id']].astype("int32")
        self.voc = data['voc2000']
        train_label = [data['label'][i] for i in data['train_id']]
        test_label = [data['label'][i] for i in data['test_id']]
        self.N, self.vocab_size = self.train_data.shape

    def __getitem__(self, index):
        topic_data = self.train_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N

def get_loader_txt(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomDataset_txt(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc

class CustomTestDataset_txt(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['data_2000'].toarray()
        self.test_data = data_all[data['test_id']].astype("int32")
        self.voc = data['voc2000']
        train_label = [data['label'][i] for i in data['train_id']]
        test_label = [data['label'][i] for i in data['test_id']]
        self.N, self.vocab_size = self.test_data.shape

    def __getitem__(self, index):
        topic_data = self.test_data[index, :]
        return np.squeeze(topic_data), 1

    def __len__(self):
        return self.N

def get_test_loader_txt(topic_data_file, batch_size=200, shuffle=True, num_workers=0):
    dataset = CustomTestDataset_txt(topic_data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc
