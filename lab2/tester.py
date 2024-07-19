# implement your testing script here
from Dataloader import *

dataloader = MIBCI2aDataset('train')
print(dataloader.labels)
print(dataloader.features)
