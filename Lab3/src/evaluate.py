import torch
import torch.nn as nn
from tqdm import tqdm
import utils

def evaluate(net, data, device):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_dice = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for image, mask in tqdm(data, desc="Evaluating"):
            image, mask = image.to(device), mask.to(device)
            
            outputs = net(image)
            
            mask = mask.squeeze(1).long()
            
            loss = criterion(outputs, mask)
            dice = utils.dice_score(outputs, mask)
            accuracy = utils.accuracy_score(outputs, mask)
            
            total_loss += loss.item()
            total_dice += dice.item()
            total_accuracy += accuracy.item()

    avg_loss = total_loss / len(data)
    avg_dice = total_dice / len(data)
    avg_accuracy = total_accuracy / len(data)
    
    return avg_loss, avg_dice, avg_accuracy