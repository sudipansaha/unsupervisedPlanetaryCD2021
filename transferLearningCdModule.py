# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:23:40 2021

@author: Sudipan
"""

import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16,vgg16_bn, vgg19_bn

import numpy as np
import scipy.io as sio
from skimage.transform import resize
from skimage import filters
from skimage import morphology
from skimage.filters import rank
import cv2 



nanVar=float('nan')


def transferLearningCd(data1,data2, model, layerNumber):
    
    ##Checking cuda availability
    useCuda = torch.cuda.is_available()
    
    ##Defining model
    
    if model == 'vgg16':
        pretrainedModel = vgg16_bn(pretrained=True)
        vgg16SubNetwork = nn.Sequential(
                    *list(pretrainedModel.features.children())[:layerNumber])
        model = vgg16SubNetwork
        #print(model)
        
 

    #print('Evaluating for CD')
    
    modelInputMean =  0
    detectedChangeMapNormalized = detectChangeGivenModel(data1, data2, model, modelInputMean, useCuda)
    return detectedChangeMapNormalized

   



def detectChangeGivenModel(preChangeImage, postChangeImage, model, modelInputMean, useCuda):    
    
    
    preChangeImageOriginalShape = preChangeImage.shape
    
    data1=np.copy(preChangeImage)  
    data2=np.copy(postChangeImage)  
    
    
  
    featurePercentileToDiscard=0
    filterNumberForOutputLayer = (model[-1].weight).shape[0]
    featureNumberToRetain=int(np.floor(filterNumberForOutputLayer*((100-featurePercentileToDiscard)/100)))
    
    
    

    eachPatch = data1.shape[0]
    
    imageSize=data1.shape
    imageSizeRow=imageSize[0]
    imageSizeCol=imageSize[1]
    
   
    if useCuda:
        model = model.cuda()
        
    model.requires_grad=False
    
    
    timeVector1Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
    timeVector2Feature=np.zeros([imageSizeRow,imageSizeCol,filterNumberForOutputLayer])
    
    
    patchToProcessDate1=data1
    patchToProcessDate2=data2
    
    patchToProcessDate1=patchToProcessDate1-modelInputMean
            
    inputToNetDate1=torch.from_numpy(patchToProcessDate1)
    inputToNetDate1=inputToNetDate1.float()
    inputToNetDate1=np.swapaxes(inputToNetDate1,0,2)
    inputToNetDate1=np.swapaxes(inputToNetDate1,1,2)
    inputToNetDate1=inputToNetDate1.unsqueeze(0)
    
    
    patchToProcessDate2=patchToProcessDate2-modelInputMean
    
    inputToNetDate2=torch.from_numpy(patchToProcessDate2)
    inputToNetDate2=inputToNetDate2.float()
    inputToNetDate2=np.swapaxes(inputToNetDate2,0,2)
    inputToNetDate2=np.swapaxes(inputToNetDate2,1,2)
    inputToNetDate2=inputToNetDate2.unsqueeze(0)
    
    if useCuda:
        inputToNetDate1 = inputToNetDate1.cuda()
        inputToNetDate2 = inputToNetDate2.cuda()
    
    
    #running model on image 1 and converting features to numpy format
    with torch.no_grad():
        obtainedFeatureVals1=model(inputToNetDate1)
    obtainedFeatureVals1=obtainedFeatureVals1.squeeze()
    obtainedFeatureVals1=obtainedFeatureVals1.data.cpu().numpy()
    
    #running model on image 2 and converting features to numpy format
    with torch.no_grad():
        obtainedFeatureVals2=model(inputToNetDate2)
    obtainedFeatureVals2=obtainedFeatureVals2.squeeze()
    obtainedFeatureVals2=obtainedFeatureVals2.data.cpu().numpy()
    #this features are in format (filterNumber, sizeRow, sizeCol)
    
    
    ##clipping values to +1 to -1 range, be careful, if network is changed, maybe we need to modify this
    obtainedFeatureVals1=np.clip(obtainedFeatureVals1,-1,+1)
    obtainedFeatureVals2=np.clip(obtainedFeatureVals2,-1,+1)
    
    
    timeVector1Feature = np.transpose(obtainedFeatureVals1,[1,2,0])
    timeVector2Feature = np.transpose(obtainedFeatureVals2,[1,2,0])

         
                                                  
                                   
    timeVectorDifferenceMatrix=timeVector1Feature-timeVector2Feature
    
    nonZeroVector=[]
    stepSizeForStdCalculation=int(imageSizeRow/2)
    for featureSelectionIter1 in range(0,imageSizeRow,stepSizeForStdCalculation):
        for featureSelectionIter2 in range(0,imageSizeCol,stepSizeForStdCalculation):
            timeVectorDifferenceSelectedRegion=timeVectorDifferenceMatrix\
                                               [featureSelectionIter1:(featureSelectionIter1+stepSizeForStdCalculation),\
                                                featureSelectionIter2:(featureSelectionIter2+stepSizeForStdCalculation),
                                                0:filterNumberForOutputLayer]
            stdVectorDifferenceSelectedRegion=np.std(timeVectorDifferenceSelectedRegion,axis=(0,1))
            featuresOrderedPerStd=np.argsort(-stdVectorDifferenceSelectedRegion)   #negated array to get argsort result in descending order
            nonZeroVectorSelectedRegion=featuresOrderedPerStd[0:featureNumberToRetain]
            nonZeroVector=np.union1d(nonZeroVector,nonZeroVectorSelectedRegion)
            
            
    modifiedTimeVector1=timeVector1Feature[:,:,nonZeroVector.astype(int)]
    modifiedTimeVector2=timeVector2Feature[:,:,nonZeroVector.astype(int)]
    
    
    ##Normalize the features (separate for both images)
    meanVectorsTime1Image=np.mean(modifiedTimeVector1,axis=(0,1))      
    stdVectorsTime1Image=np.std(modifiedTimeVector1,axis=(0,1))
    normalizedModifiedTimeVector1=(modifiedTimeVector1-meanVectorsTime1Image)/stdVectorsTime1Image
    
    meanVectorsTime2Image=np.mean(modifiedTimeVector2,axis=(0,1))      
    stdVectorsTime2Image=np.std(modifiedTimeVector2,axis=(0,1))
    normalizedModifiedTimeVector2=(modifiedTimeVector2-meanVectorsTime2Image)/stdVectorsTime2Image
    
   
    timeVector1FeatureAggregated=np.copy(normalizedModifiedTimeVector1)
    timeVector2FeatureAggregated=np.copy(normalizedModifiedTimeVector2)
    
    
    del obtainedFeatureVals1, obtainedFeatureVals2, timeVector1Feature, timeVector2Feature, inputToNetDate1, inputToNetDate2 
    
    
    timeVector1FeatureMaxValues = np.amax(timeVector1FeatureAggregated,axis=(0,1))
    timeVector2FeatureMaxValues = np.amax(timeVector2FeatureAggregated,axis=(0,1))
    

    changeIndicator = np.mean(np.abs(timeVector1FeatureMaxValues-timeVector2FeatureMaxValues))
    
    

    
    
    return changeIndicator
      
        
   
    
        