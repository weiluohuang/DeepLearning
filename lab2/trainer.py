import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import model.SCCNet as SCCNet
import Dataloader
import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SD_trainset = Dataloader.MIBCI2aDataset('train', 'SD')
# LOSO_trainset = Dataloader.MIBCI2aDataset('train', 'LOSO')
# FT_trainset = Dataloader.MIBCI2aDataset('train', 'FT')

SD_trainloader = DataLoader(SD_trainset, batch_size=64,shuffle=True)
# LOSO_trainloader = DataLoader(SD_trainset, batch_size=64,shuffle=True)
# FT_trainloader = DataLoader(SD_trainset, batch_size=64,shuffle=True)

model1 = SCCNet.SCCNet().to(device)
n_epochs = 5000
loss_history = []
optimizer = optim.Adam(model1.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
model1.train()
lowest_loss = 2
for epoch in range(1, n_epochs+1):
    for batch_idx, (features, labels) in enumerate(SD_trainloader):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model1(features.to(torch.float32))
        loss = criterion(output, labels.to(torch.int64))
        if loss < lowest_loss:
            lowest_loss = loss
            torch.save(model1.state_dict(), 'SD_best.pth')
            print("loss update : ", loss.cpu().detach().numpy())
        loss_history.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
    if not epoch%100:
        print("now in epoch : ", epoch)
torch.save(model1.state_dict(), 'SD.pth')
# plt.plot(loss_history)
# plt.show()