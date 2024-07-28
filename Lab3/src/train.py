import torch
import argparse
from models import unet, resnet34_unet
import oxford_pet
import torch.optim as optim
import torch.nn as nn
import utils
from tqdm import tqdm
from evaluate import evaluate
import json

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader = oxford_pet.load_dataset(args.data_path, "train", args.batch_size)
    valloader = oxford_pet.load_dataset(args.data_path, "valid", args.batch_size)
    
    model = resnet34_unet.ResNet34_UNet().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_accuracies = []
    val_accuracies = []
    
    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_accuracy = 0
        for i, (image, mask) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch}/{args.epochs}")):
            image, mask = image.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(image)
            
            mask = mask.squeeze(1).long()
            
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                accuracy = utils.accuracy_score(outputs, mask)
            
            epoch_accuracy += accuracy.item()

        avg_accuracy = epoch_accuracy / len(trainloader)
        
        train_accuracies.append(avg_accuracy)
        
        # Validation phase
        _, _, val_accuracy = evaluate(model, valloader, device)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch}/{args.epochs}, Train Accuracy: {avg_accuracy:.3f}, Val Accuracy: {val_accuracy:.3f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f'./Lab3/saved_models/ResNet34_UNet_{int(best_accuracy*100)}.pth')
            print(f"New best model saved with Accuracy: {best_accuracy:.3f}")

    print(f"Training completed. Best Accuracy: {best_accuracy:.3f}")
    
    # Save training history
    history = {
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }
    
    with open(f'ResNet34_UNet_history.json', 'w') as f:
        json.dump(history, f)
    
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on image and target mask')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)

# python ./Lab3/src/train.py --data_path ./Lab3/dataset/oxford-iiit-pet/ --epochs 150 --batch_size 8 --learning-rate 1e-2