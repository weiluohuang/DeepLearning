import torch
import argparse
from models import unet #, resnet34_unet
import oxford_pet
import torch.optim as optim
import torch.nn as nn

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader = oxford_pet.load_dataset(args.data_path, "train", args.batch_size)
    
    model = unet.UNet(3, 1).to(device)  # 假設輸入是3通道，輸出是1通道
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()  # 假設這是二元分割任務
    
    model.train()
    lowest_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        for batch_idx, (images, masks) in enumerate(trainloader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images.to(torch.float32))
            loss = criterion(outputs, masks.to(torch.float32))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # 計算平均 epoch loss
        avg_loss = epoch_loss / len(trainloader)
        
        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < lowest_loss:
            lowest_loss = avg_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"New best model saved with loss: {lowest_loss:.4f}")
    
    # 保存最終模型
    torch.save(model.state_dict(), 'final_unet_model.pth')
    print("Training completed. Final model saved.")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)

# python .\Lab3\src\train.py --data_path ./Lab3/dataset/oxford-iiit-pet/ --epochs 1000 --batch_size 64 --learning-rate 0.01