# -*- coding: utf-8 -*-
"""
Patch-level unsupervised planetary change detection
"""
import os
import glob
import sys
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transferLearningCdModule import transferLearningCd
import cv2 

import random

import time


nanVar=float('nan')

##Defining data paths
dataDirectoryPath = './planetaryCDDataset/hirise_before_after_grayscale/garni_029213_029780_siamese/grayscale/'

##Extracting which images are changed and which are unchanged
changeLabelDirectoryPath = './planetaryCDDataset/hirise_rsl/rsl_coregistered/garni_029213_029780/garni_029213_029780_signdiff_lcn/change'
unchangedLabelDirectoyPath = './planetaryCDDataset/hirise_rsl/rsl_coregistered/garni_029213_029780/garni_029213_029780_signdiff_lcn/no_change'
changedImages = glob.glob(changeLabelDirectoryPath+'/*.jpg')
for changedImageIter in range(len(changedImages)):
    thisIterChangedImage = changedImages[changedImageIter]
    thisIterChangedImage = (os.path.basename(thisIterChangedImage).rsplit(".",1))[0]
    changedImages[changedImageIter] = thisIterChangedImage

unchangedImages = glob.glob(unchangedLabelDirectoyPath+'/*.jpg')
for unchangedImageIter in range(len(unchangedImages)):
    thisIterUnchangedImage = unchangedImages[unchangedImageIter]
    thisIterUnchangedImage = (os.path.basename(thisIterUnchangedImage).rsplit(".",1))[0]
    unchangedImages[unchangedImageIter] = thisIterUnchangedImage

##Defining parameters
topPercentToSaturate=1

### Paramsters related to the CNN model
modelInputMean=0
useCuda = torch.cuda.is_available()   


##setting manual seeds
manualSeed=40
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
np.random.seed(manualSeed)









class HiriseDatasetLoader(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, transform=None):
        """
        Argumets:
            
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # basics
        self.transform = transform
        
        # load images
        self.allPrechangeImages = glob.glob(dataDirectoryPath+'/*before*.jpg')
        self.numImage = len(self.allPrechangeImages)
      
        
    def __len__(self):
        
        return self.numImage

    def __getitem__(self, idx):
               
        preChangeImagePath = self.allPrechangeImages[idx]
        preChangeImageBaseFileName = os.path.basename(preChangeImagePath)
        preChangeImageIndexBefore = preChangeImageBaseFileName.find('before')
        imageId = preChangeImageBaseFileName[0:(preChangeImageIndexBefore-3)]
        postChangeImagePath = 'after'.join(preChangeImagePath.rsplit('before', 1)) 
        I1 = self.reshapeForTorch(cv2.imread(preChangeImagePath))/255. ##Here data was uint in range 0-255
        I2 = self.reshapeForTorch(cv2.imread(postChangeImagePath))/255.  ##Here data was uint in range 0-255
        
        if imageId in changedImages:
            label = 1  
        elif imageId in unchangedImages:
            label = 0
        else:
            sys.exit('Image does not have an associated label')
        sample = {'I1': I1, 'I2': I2, 'label': label, 'preChangeImagePath': preChangeImagePath, 'postChangeImagePath': postChangeImagePath}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    def reshapeForTorch(self,I):
        """Transpose image for PyTorch coordinates."""
        out = I.transpose((2, 0, 1))
        return torch.from_numpy(out)



class Normalize(object):
    """substitute for transforms.normalize."""

#     def __init__(self):
#         return
    def __init__(self, mean1, std1, mean2, std2):
        super().__init__()
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
    
    def __call__(self, sample):        
        I1, I2, label, preChangeImagePath, postChangeImagePath  = sample['I1'], sample['I2'], sample['label'], sample['preChangeImagePath'], sample['postChangeImagePath']
        norm1 = transforms.Normalize(self.mean1, self.std1)
        norm2 = transforms.Normalize(self.mean2, self.std2)
               
        I1 = norm1(I1)
        I2 = norm2(I2)

        return {'I1': I1, 'I2': I2, 'label': label, 'preChangeImagePath': preChangeImagePath, 'postChangeImagePath': postChangeImagePath}






dataTransform = transforms.Compose([
                                   Normalize((0.5, 0.5, 0.5), (0.15, 0.15, 0.15),(0.5, 0.5, 0.5), (0.15, 0.15, 0.15))
                                   #Normalize((0, 0, 0), (1, 1, 1),(0, 0, 0), (1, 1, 1))
                                   ])
testDataset = HiriseDatasetLoader(dataTransform)



testLoader = torch.utils.data.DataLoader(dataset=testDataset,batch_size=1,shuffle=False) 


correctPrediction = 0
wrongPrediction = 0
truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0


for batchStep, batchData in enumerate(testLoader):
    data1 = batchData['I1'].float()
    data2 = batchData['I2'].float()
    
    preChangeImage = np.transpose(np.squeeze(data1),[1,2,0])
    postChangeImage = np.transpose(np.squeeze(data2),[1,2,0])
        
    label = batchData['label'].float().cuda()
        
             
  
    changeIndicator = transferLearningCd(preChangeImage,postChangeImage,'vgg16',18)
    
    prediction = changeIndicator>0.2860
       
    if label == 1:
        if prediction==1:
            correctPrediction = correctPrediction+1
            truePositive = truePositive+1   
        if prediction==0:
            wrongPrediction = wrongPrediction+1
            falseNegative = falseNegative+1
    if label == 0:
        if prediction==1:
            wrongPrediction = wrongPrediction+1
            falsePositive = falsePositive+1
        if prediction==0:
            correctPrediction = correctPrediction+1
            trueNegative = trueNegative+1
            

            
accuracy = (truePositive+trueNegative)/(truePositive+trueNegative+falsePositive+falseNegative)           
print('Correct prediction: '+str(correctPrediction))
print('Wrong prediction: '+str(wrongPrediction))
print('Accuracy: '+str(accuracy))

print('True positive: '+str(truePositive))
print('True negative: '+str(trueNegative))
print('False positive: '+str(falsePositive))
print('False negative: '+str(falseNegative))




            
            
            
            
           


    
    
    
    
    







    
    
    
    
    






    



