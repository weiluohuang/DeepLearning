import torch
import argparse
from models import unet #, resnet34_unet
import oxford_pet
import torch.optim as optim
import torch.nn as nn
import utils

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader = oxford_pet.load_dataset(args.data_path, "train", args.batch_size)
    
    model = unet.UNet(3, 2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        for i, (image, mask) in enumerate(trainloader):
            image, mask = image.to(device), mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(image)
            
            mask = mask.squeeze(1).long()
            
            loss = criterion(outputs, mask)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                dice = utils.dice_score(outputs, mask)
            
            if i is len(trainloader):
                print(f"Epoch {epoch}/{args.epochs}, Loss: {loss.item():.4f}")

            if dice.item() > best_dice:
                best_dice = dice.item()
                torch.save(model.state_dict(), f'./Lab3/saved_models/unet{best_dice*100:.0f}.pth')
                print(f"New best model saved with Dice score: {best_dice:.3f}")

    print(f"Training completed. Final model saved. Best Dice score: {best_dice:.3f}")

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

# python ./Lab3/src/train.py --data_path ./Lab3/dataset/oxford-iiit-pet/ --epochs 100 --batch_size 16 --learning-rate 1e-1