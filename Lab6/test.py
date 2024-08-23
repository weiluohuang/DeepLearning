import os
import torch
from torchvision.utils import make_grid
from diffusers import DDPMScheduler
from dataloader import get_loader
import torchvision.transforms as transforms
from evaluator import evaluation_model
from train import ClassConditionedUnet

def eva(mode='test'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    noise_scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2')

    model = ClassConditionedUnet().to(device)
    checkpoint = torch.load('./pth/11.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    evaluator = evaluation_model()
    os.makedirs("imgs", exist_ok=True)

    test_loader = get_loader(mode)
    img_grid = []
    accuracys = []

    for idx, y in enumerate(test_loader):
        y = y.to(device)
        x = torch.randn(1, 3, 64, 64).to(device)
        process = []
        for i, t in enumerate(noise_scheduler.timesteps):

            with torch.no_grad():
                residual = model(x, t, y.squeeze(1))

            x = noise_scheduler.step(residual, t, x).prev_sample
            if i % 100 == 0:
                process.append((x * 0.5 + 0.5).detach().cpu().squeeze(0))

        accuracy = evaluator.eval(x, y.squeeze(1))
        accuracys.append(accuracy)
        print(idx, " acc: ", accuracy)
        
        img = (x * 0.5 + 0.5).detach().cpu().squeeze(0)
        process.append(img)
        tensor_images = torch.stack(process)
        row_image = make_grid(tensor_images, nrow=len(process))
        row_image_pil = transforms.ToPILImage()(row_image)
        row_image_pil.save(f'imgs/process_{mode}_{idx}.png')

        img_grid.append(img)
    grid_image = make_grid(img_grid, nrow=8)
    grid_image_pil = transforms.ToPILImage()(grid_image)
    grid_image_pil.save(f'imgs/grid_{mode}.png')
    return sum(accuracys)/len(accuracys)


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Test Accuracy: ', eva('test'))
    print('New Test Accuracy: ', eva('new_test'))