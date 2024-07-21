import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import model.SCCNet as SCCNet
import Dataloader

SD_trainset = Dataloader.MIBCI2aDataset('train', 'SD')
trainloader = DataLoader(SD_trainset, batch_size=16,shuffle=True)

model1 = SCCNet.SCCNet()
n_epochs = 10000
for epoch in range(1, n_epochs+1):
    optimizer = optim.Adam(model1.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model1.train()
    for batch_idx, (features, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model1(features)
        loss = criterion(output, labels)
        if not epoch%10:
            print(loss) 
        loss.backward()
        optimizer.step()