import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
import torch.nn.functional as F
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        checkpoint = torch.load(load_ckpt_path)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('transformer.'):
                new_k = k.replace('transformer.', '')
                new_state_dict[new_k] = v
        self.transformer.load_state_dict(new_state_dict, strict=True)

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        z_q, indices, _ = self.vqgan.encode(x)
        indices = indices.view(z_q.size(0), 16, 16)
        return z_q, indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        def linear(ratio):
            return 1 - ratio

        def cosine(ratio):
            return math.cos(math.pi / 2 * ratio)

        def square(ratio):
            return 1 - ratio**2
        
        if mode == "linear":
            return linear
        elif mode == "cosine":
            return cosine
        elif mode == "square":
            return square
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        _, z_indices = self.encode_to_z(x) #ground truth
    
        batch_size, height, width = z_indices.shape
        seq_length = height * width
        z_indices = z_indices.view(batch_size, -1)

        # generate random mask
        ratio = torch.rand(1).item()
        mask_ratio = self.gamma(ratio)
        num_masks = int(seq_length * mask_ratio)
        
        mask = torch.zeros(batch_size, seq_length, device=z_indices.device).bool()
        mask[:, :num_masks] = True

        for i in range(batch_size):
            mask[i] = mask[i, torch.randperm(seq_length)]
        
        masked_indices = z_indices.clone()
        masked_indices[mask] = self.mask_token_id
        
        logits = self.transformer(masked_indices)

        return logits, z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask, ratio):
        masked_indices = z_indices.clone().view(1, 256)
        masked_indices[mask] = self.mask_token_id
        logits = self.transformer(masked_indices)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        probs = F.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(probs, dim=-1)

        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))  # gumbel noise
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        confidence_masked = confidence.masked_fill(~mask, -float('inf'))
        _, indices = torch.sort(confidence_masked, descending=True)
        num_tokens_to_keep = int((1 - ratio) * mask.sum())
        indices_to_unmask = indices[0, :num_tokens_to_keep]

        mask_bc = mask.clone()
        mask_bc[0, indices_to_unmask] = False
        
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
