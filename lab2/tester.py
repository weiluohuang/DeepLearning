# implement your testing script here
from Dataloader import *

dataloader = MIBCI2aDataset('finetune','FT')
print(dataloader.labels.shape)
print(dataloader.features.shape)