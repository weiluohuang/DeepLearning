import torch
import os
from diffusers import DDPMScheduler, UNet2DModel
from dataloader import get_loader
from tqdm import tqdm

class ClassConditionedUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size = 64,
            block_out_channels = (128, 128, 256, 256, 512, 512), 
            down_block_types=('DownBlock2D', 'DownBlock2D', 'DownBlock2D','DownBlock2D','DownBlock2D', 'AttnDownBlock2D'),
            up_block_types=('AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D')
        )
        self.model.class_embedding = torch.nn.Linear(24, 512)

    def forward(self, x, t, class_label):
        return self.model(x, t, class_label).sample

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    noise_scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2')
    train_loader = get_loader('train')
    n_epochs = 100
    model = ClassConditionedUnet().to(device)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)


    for epoch in range(1, n_epochs+1):
        losses = []
        for x, y in tqdm(train_loader, desc="Training", ncols=50):

            x = x.to(device)
            y = y.to(device).squeeze(1)

            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            pred = model(noisy_x, timesteps, y)

            loss = loss_fn(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        epoch_loss = sum(losses)/len(losses)

        print(f'Epoch {epoch}/{n_epochs}, loss: {epoch_loss}')
        os.makedirs("pth", exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(),}, f'./pth/{epoch}.pth')