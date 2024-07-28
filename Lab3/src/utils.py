import torch

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