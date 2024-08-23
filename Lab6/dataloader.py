import torch
import os
import json
import torch.utils
import torch.utils.data.dataloader
import torchvision
import PIL.Image
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class dataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        assert mode in {'train', 'test', 'new_test'}
        self.mode = mode
        with open('./objects.json', 'r') as file:
            self.dict = json.load(file)
        if mode == 'train':
            self.image, self.label = self.load_json()
        else:
            self.label = self.load_json()
        self.mlb = MultiLabelBinarizer().fit([(x,) for x in range(24)])

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path = os.path.join('./iclevr', self.image[idx])
            transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img = transform(PIL.Image.open(img_path).convert('RGB'))
            label = self.mlb.transform([self.label[idx]])
            return img, torch.from_numpy(label).float()
        else:
            label = self.mlb.transform([self.label[idx]])
            return torch.from_numpy(label).float()
        
    def load_json(self):
        if self.mode == 'train':
            with open('./train.json', 'r') as file:
                data = json.load(file)
                img = []
                label = []
                for i in data:
                    img.append(i)
                    label_names = data[i]
                    labels = [self.dict[x] for x in label_names]
                    label.append(np.array(labels))
                    
            return img, label
        
        else:
            with open('./test.json'if self.mode == 'test' else './new_test.json', 'r') as file:
                data = json.load(file)
                label = []
                for i in data:
                    label.append(np.array([self.dict[x] for x in i]))
            return label
        
    
def get_loader(mode='train'):
    return torch.utils.data.DataLoader(dataset(mode), batch_size=8 if mode == 'train' else 1, shuffle=(mode == 'train'), num_workers=(mode == 'train'))