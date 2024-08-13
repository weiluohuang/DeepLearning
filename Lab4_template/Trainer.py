import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10
import torch.backends.cudnn as cudnn

torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = False
cudnn.deterministic = True

random.seed(0) 
np.random.seed(0)
torch.manual_seed(0)

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.epoch = current_epoch
        if self.type == 'Cyclical':
            self.beta, self.L = self.frange_cycle_linear(args.num_epoch + 1, n_cycle=self.cycle, ratio=self.ratio)
        elif self.type == 'Monotonic':
            self.beta, self.L = self.frange_cycle_linear(args.num_epoch + 1, n_cycle=1, ratio=self.ratio)
            # for i in range(args.num_epoch + 1):
            #     if self.L[i] != 1.0:
            #         continue
            #     for j in range(i + 1, args.num_epoch + 1):
            #         self.L[j] = 1.0
            #     break
        else:
            self.L = np.ones(args.num_epoch + 1) * self.ratio
            self.beta = self.L[0]
        print("kl_annealing schedule:", self.L)
        
    def update(self):
        self.epoch += 1
        self.beta = self.L[self.epoch]
    
    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):  
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop - start)/(period * ratio)
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L[0], L
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[], gamma=0.5)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        self.train_losses = []
        self.val_losses = []
        self.tfr_values = []
        self.PSNR = []
        
        
    def forward(self, img, label):
        pass
    
    def plot_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(self.args.save_root, 'loss_curve.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.tfr_values)
        plt.xlabel('Epoch')
        plt.ylabel('Teacher Forcing Ratio')
        plt.title('Teacher Forcing Ratio over Epochs')
        plt.savefig(os.path.join(self.args.save_root, 'tfr_curve.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.PSNR, label='PSNR')
        plt.xlabel('Frame')
        plt.ylabel('PSNR')
        plt.title(f'Validation PSNR ({self.args.kl_anneal_type})')
        plt.legend()
        plt.savefig(f'./{self.args.save_root}/{self.args.kl_anneal_type}_PSNR_per_frame.png')
        plt.close()

    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            epoch_loss = 0.0
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                epoch_loss += loss.item()

                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.2f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.2f}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

            self.train_losses.append(epoch_loss / len(train_loader))
            self.tfr_values.append(self.tfr)
                
            val_loss = self.eval()
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"ep{self.current_epoch}L{int(val_loss*1000)}.ckpt"))
            self.val_losses.append(val_loss)
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_loss = 0.0
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            total_loss += loss.item()
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        return total_loss / len(val_loader)
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        self.optim.zero_grad()
        
        img = img.permute(1, 0, 2, 3, 4) # [T, B, C, H, W]
        label = label.permute(1, 0, 2, 3, 4)
        output = img[0]

        MSE_loss = 0.0
        kl_loss = 0.0
        for i in range(1, self.train_vi_len):
            if adapt_TeacherForcing:
                output = img[i-1]
            # output = img[i-1] * self.tfr + output * (1 - self.tfr)
            
            label_feature = self.label_transformation(label[i])
            frame_feature = self.frame_transformation(output)
            
            z, mu, logvar = self.Gaussian_Predictor(frame_feature, label_feature)
            
            output = self.Generator(self.Decoder_Fusion(frame_feature, label_feature, z))

            MSE_loss += self.mse_criterion(output, img[i])
            kl_loss += kl_criterion(mu, logvar, batch_size = self.batch_size)

        beta = torch.tensor(self.kl_annealing.get_beta()).to(self.args.device)
        loss = MSE_loss + beta * kl_loss
        
        loss.backward()
        self.optimizer_step()

        return loss
    
    def val_one_step(self, img, label):
        img = img.permute(1, 0, 2, 3, 4)
        label = label.permute(1, 0, 2, 3, 4)
        decoded_frame_list = [img[0].cpu()]
        output = img[0]
        self.PSNR = []

        MSE_loss = 0.0
        for i in range(1, self.val_vi_len):
            label_feat = self.label_transformation(label[i])
            frame_feature = self.frame_transformation(output)
            
            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W, device=self.args.device, dtype=torch.float)
              
            output = self.Generator(self.Decoder_Fusion(frame_feature, label_feat, z))
            decoded_frame_list.append(output.cpu())
            self.PSNR.append(Generate_PSNR(output, img[i]).detach().cpu())
            MSE_loss += self.mse_criterion(output, img[i])
        
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        self.make_gif(generated_frame[0], os.path.join(self.args.save_root, f'{self.args.kl_anneal_type}.gif'))
        self.plot_curves()
        
        return MSE_loss
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde and self.tfr > 0:
            self.tfr -= self.tfr_d_step
        if self.tfr < 0:
            self.tfr = 0
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.0e}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoints(self):
        if self.args.ckpt_path != None:
            checkpoints = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoints['state_dict'], strict=True) 
            self.args.lr = checkpoints['lr']
            self.tfr = checkpoints['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[7,14,21], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoints['last_epoch'])
            self.current_epoch = checkpoints['last_epoch'] + 1

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoints()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=1)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoints every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)

# python ./Lab4_template/Trainer.py --DR ./LAB4_Dataset --save_root ./Lab4_template/checkpoints/ --per_save 1 --num_epoch 100 --fast_train --fast_partial 0.4 --fast_train_epoch 10 --kl_anneal_cycle 20
# set gamma=0.2, you can get ep99L840.ckpt

# python ./Lab4_template/Trainer.py --DR ./LAB4_Dataset --save_root ./Lab4_template/checkpoints/ --per_save 1 --num_epoch 200 --kl_anneal_cycle 40 --ckpt_path ./Lab4_template/checkpoints/ep99L840.ckpt
# self.current_epoch = checkpoints['last_epoch'] + 1
# milestones=[7,14,21] you will get ep125L289.ckpt