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
        # implement the getFeatures method
        return self.load_and_concatenate(filePath)

    def _getLabels(self, filePath):
        # implement the getLabels method
        return self.load_and_concatenate(filePath)

    def __init__(self, mode):
        # remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        if mode == 'train':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath='./lab2/dataset/LOSO_train/features/')
            self.labels = self._getLabels(filePath='./lab2/dataset/LOSO_train/labels/')
        if mode == 'finetune':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath='./lab2/dataset/FT/features/')
            self.labels = self._getLabels(filePath='./lab2/dataset/FT/labels/')
        if mode == 'test':
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath='./lab2/dataset/SD_test/features/')
            self.labels = self._getLabels(filePath='./lab2/dataset/SD_test/labels/')

    def __len__(self):
        # implement the len method
        pass

    def __getitem__(self, idx):
        # implement the getitem method
        pass