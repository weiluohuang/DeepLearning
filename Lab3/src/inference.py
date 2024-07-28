import argparse
import torch
from models import unet, resnet34_unet
import oxford_pet
import evaluate

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == './Lab3/saved_models/UNet.pth':
        model = unet.UNet(3, 2).to(device)
    else:
        model = resnet34_unet.ResNet34_UNet().to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    test_loader = oxford_pet.load_dataset(args.data_path, "test", args.batch_size)
    loss_score, dice_score, accuracy_score = evaluate.evaluate(model, test_loader, device)
    print(f"Test Loss: {loss_score:.4f}")
    print(f"Test Dice Score: {dice_score:.4f}")
    print(f"Test Accuracy: {accuracy_score:.4f}")

    # python ./Lab3/src/inference.py --data_path ./Lab3/dataset/oxford-iiit-pet/ --batch_size 8 --model ./Lab3/saved_models/ResNet34_UNet.pth