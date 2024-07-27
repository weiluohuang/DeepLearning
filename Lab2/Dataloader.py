import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

class MIBCI2aDataset(Dataset):

    def load_and_concatenate(self, folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        arrays = []
        for file in files:
            file_path = os.path.join(folder_path, file)
            array = np.load(file_path)
            arrays.append(array)
        concatenated_array = np.concatenate(arrays, axis=0) 
        return concatenated_array

    def _getFeatures(self, filePath):
        features = self.load_and_concatenate(filePath)
        return torch.from_numpy(features).float()

    def _getLabels(self, filePath):
        labels = self.load_and_concatenate(filePath)
        return torch.from_numpy(labels).long()

    def __init__(self, mode, experiments):
        assert mode in ['train', 'test', 'finetune']
        assert experiments in ['SD', 'LOSO']
        if mode == 'train':
            if experiments == 'SD':
                self.features = self._getFeatures(filePath='./lab2/dataset/SD_train/features/')
                self.labels = self._getLabels(filePath='./lab2/dataset/SD_train/labels/')
            else:
                self.features = self._getFeatures(filePath='./lab2/dataset/LOSO_train/features/')
                self.labels = self._getLabels(filePath='./lab2/dataset/LOSO_train/labels/')
        if mode == 'test':
            if experiments == 'SD':
                self.features = self._getFeatures(filePath='./lab2/dataset/SD_test/features/')
                self.labels = self._getLabels(filePath='./lab2/dataset/SD_test/labels/')
            else:
                self.features = self._getFeatures(filePath='./lab2/dataset/LOSO_test/features/')
                self.labels = self._getLabels(filePath='./lab2/dataset/LOSO_test/labels/')
        if mode == 'finetune':
            self.features = self._getFeatures(filePath='./lab2/dataset/FT/features/')
            self.labels = self._getLabels(filePath='./lab2/dataset/FT/labels/')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# dataset = MIBCI2aDataset("test", "SD")
# loader = DataLoader(dataset, 1, False)
# for labels in enumerate(loader):
#     print(labels)
#     break