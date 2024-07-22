import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import model.SCCNet as SCCNet
import Dataloader
import matplotlib.pyplot as plt
# plt.figure()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SD_trainset = Dataloader.MIBCI2aDataset('train', 'LOSO')
SD_trainloader = DataLoader(SD_trainset, batch_size=32,shuffle=True)

model1 = SCCNet.SCCNet(Nu=44, Nc=20, Nt=16).to(device)
n_epochs = 10000
# loss_history = []
optimizer = optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-4)
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
            torch.save(model1.state_dict(), 'LOSO_best.pth')
            print("loss update : ", loss.cpu().detach().numpy())
        # loss_history.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
    if not epoch%1000:
        print("now in epoch : ", epoch)
torch.save(model1.state_dict(), 'LOSO.pth')

# plt.plot(loss_history)
# plt.show()