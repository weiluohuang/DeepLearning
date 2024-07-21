import torch
import numpy as np
import os

class MIBCI2aDataset(torch.utils.data.Dataset):

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
        return self.load_and_concatenate(filePath)

    def _getLabels(self, filePath):
        return self.load_and_concatenate(filePath)

    def __init__(self, mode, experiments):
        assert mode in ['train', 'test', 'finetune']
        assert experiments in ['SD', 'LOSO', 'FT']
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
        # implement the len method
        pass

    def __getitem__(self, idx):
        # implement the getitem method
        pass