import torch
import json
import matplotlib.pyplot as plt

def plot_comparison(model1_history, model2_history, model1_name, model2_name):
    with open(model1_history, 'r') as f:
        history1 = json.load(f)
    
    with open(model2_history, 'r') as f:
        history2 = json.load(f)

    plt.figure(figsize=(12,5))
    
    # Plot training accuracies
    plt.subplot(1, 2, 1)
    plt.plot(history1['train_accuracy'], label=model1_name)
    plt.plot(history2['train_accuracy'], label=model2_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # Plot validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history1['val_accuracy'], label=model1_name)
    plt.plot(history2['val_accuracy'], label=model2_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def dice_score(pred_mask, gt_mask):
    pred_mask = torch.argmax(pred_mask, dim=1)
    
    gt_mask = gt_mask.squeeze(1) # [batch_size, height, width]
    
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()
    
    intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
    union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
    
    dice = 2. * intersection / union
    return dice.mean()

def accuracy_score(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float()
    return correct.sum() / correct.numel()

if __name__ == "__main__":
    plot_comparison('UNet_history.json', 'ResNet34_UNet_history.json', 'UNet', 'ResNet34-UNet')