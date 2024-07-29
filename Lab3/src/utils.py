import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from models import unet, resnet34_unet
import oxford_pet

def visualize_predictions(model, dataloader, device, num_samples):
    model.eval()
    
    with torch.no_grad():
        for i, (image, true_mask) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            image, true_mask = image.to(device), true_mask.to(device)
            
            output = model(image)
            pred_mask = torch.argmax(output, dim=1)
            
            image = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            true_mask = true_mask.cpu().squeeze().numpy()
            pred_mask = pred_mask.cpu().squeeze().numpy()
            
            image = (image - image.min()) / (image.max() - image.min())
            
            fig, axs = plt.subplots(1, 3, figsize=(7, 3))
            
            axs[0].imshow(image)
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            axs[1].imshow(image)
            axs[1].imshow(true_mask, alpha=0.3, cmap='jet')
            axs[1].set_title('True Mask')
            axs[1].axis('off')
            
            axs[2].imshow(image)
            axs[2].imshow(pred_mask, alpha=0.3, cmap='jet')
            axs[2].set_title('Predicted Mask')
            axs[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'./Lab3/visual/segmentation_visualization_{i}.png')
            plt.close()

def plot_comparison(model1_history, model2_history, model3_history, model4_history, model1_name, model2_name, model3_name, model4_name):
    with open(model1_history, 'r') as f:
        history1 = json.load(f)
    
    with open(model2_history, 'r') as f:
        history2 = json.load(f)

    with open(model3_history, 'r') as f:
        history3 = json.load(f)

    with open(model4_history, 'r') as f:
        history4 = json.load(f)
    
    plt.figure(figsize=(12,5))
    
    # Plot training accuracies
    plt.subplot(1, 2, 1)
    plt.plot(history1['train_accuracy'], label=model1_name)
    plt.plot(history2['train_accuracy'], label=model2_name)
    plt.plot(history3['train_accuracy'], label=model3_name)
    plt.plot(history4['train_accuracy'], label=model4_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # Plot validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history1['val_accuracy'], label=model1_name)
    plt.plot(history2['val_accuracy'], label=model2_name)
    plt.plot(history3['val_accuracy'], label=model3_name)
    plt.plot(history4['val_accuracy'], label=model4_name)
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
    plot_comparison('UNet_SGD.json', 'UNet_Adam.json', 'Res_SGD.json', 'Res_Adam.json', 'UNet_SGD', 'UNet_Adam', 'ResNet34-UNet_SGD', 'ResNet34-UNet_Adam')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = unet.UNet(3, 2).to(device)
    # model.load_state_dict(torch.load("./Lab3/saved_models/UNet.pth", map_location=device))
    # loader = oxford_pet.load_dataset("./Lab3/dataset/oxford-iiit-pet/", "test", 1)
    # visualize_predictions(model, loader, device, 10)