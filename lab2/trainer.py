import sys
import os
import model.SCCNet as SCCNet
import Dataloader

SD_train = Dataloader.MIBCI2aDataset('train', 'SD')
SD_test = Dataloader.MIBCI2aDataset('test', 'SD')
model1 = SCCNet.SCCNet(Nu=3,C=22,Nc=3,Nt=1)

print(model1.forward(SD_train))