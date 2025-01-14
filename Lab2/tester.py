import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import model.SCCNet
import Dataloader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
SD_testset = Dataloader.MIBCI2aDataset('test', 'LOSO')
SD_testloader = DataLoader(SD_testset, batch_size=64, shuffle=False)

model = model.SCCNet.SCCNet(Nu=22, Nc=20, Nt=16).to(device)
model.load_state_dict(torch.load('FT72.pth')) #macos: map_location=torch.device('cpu')
model.eval()

criterion = nn.CrossEntropyLoss()
correct = 0
total = 0
total_loss = 0

with torch.no_grad():
    for batch_idx, (features, labels) in enumerate(SD_testloader):
        features, labels = features.to(device), labels.to(device)
        outputs = model(features.to(torch.float32))
        loss = criterion(outputs, labels.to(torch.int64))
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_loss = total_loss / len(SD_testloader)
accuracy = 100 * correct / total

print(f'Test Loss: {avg_loss:.4f}')
print(f'Test Accuracy: {accuracy:.2f}%')
