from __future__ import print_function, division 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.autograd import Variable 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
#from skimage import io, transform 
import pandas as pd 
import torchvision 
from torchvision import transforms, datasets, models 
import matplotlib.pyplot as plt 
import os 
import cv2 
import torch 
import argparse
import csv

from DatasetGenerator import DatasetGenerator 
from model import *
from utils import *

parser = argparse.ArgumentParser(description='AG-CNN')

parser.add_argument('--use', type=str, default='train',
                    help='use for what (train or test)')
parser.add_argument('--branch_type', type=str, default='global',
                    help='training branch (global, local or fusion)')
parser.add_argument('--model', type=str, default='resnet50',
                    help='basic model (resneat50 or densenet121)')

args = parser.parse_args()

print(args)
#-------------------------------------------------------------------------------- 
       
def epochTrain (epochID, model, dataLoader, optimizer, loss, test_model):
        
    model.train()
    
    for batchID, (input, target) in enumerate (dataLoader):
        print(args.branch_type+' Training...'+'epoch:'+str(epochID)+'||'+'train:'+str(batchID))

        target = target.cuda(async = True)
        
        
        if args.branch_type == 'local':
            output, heatmap, pool = test_model(torch.autograd.Variable(input.cuda(async = True)))
            input = heatmap_crop_origin(heatmap.cpu(), input)

        if args.branch_type == 'fusion':
            test_model1, test_model2 = test_model
            output, heatmap, pool1 = test_model1(torch.autograd.Variable(input.cuda(async = True)))
            inputs = heatmap_crop_origin(heatmap.cpu(), input)
            output, heatmap, pool2 = test_model2(torch.autograd.Variable(inputs.cuda(async = True)))
            pool1 = pool1.view(pool1.size(0), -1)
            pool2 = pool2.view(pool2.size(0), -1)
            input = torch.cat((pool1.cpu(), pool2.cpu()), dim=1)

        input = input.cuda(async = True)
        
        varInput = torch.autograd.Variable(input)
        varTarget = torch.autograd.Variable(target)         
        varOutput, heatmap, pool = model(varInput)
            
        lossvalue = loss(varOutput, varTarget)
                       
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
            
#-------------------------------------------------------------------------------- 
        
def epochVal (epochID, model, dataLoader, optimizer, loss, test_model):
        
    model.eval ()
        
    lossVal = 0
    lossValNorm = 0
        
    losstensorMean = 0
        
    for i, (input, target) in enumerate (dataLoader):
        print(args.branch_type+' Valing...''epoch:'+str(epochID)+'||'+'eval:'+str(i))
        target = target.cuda(async = True)

        if args.branch_type == 'local':
            output, heatmap, pool = test_model(torch.autograd.Variable(input.cuda(async = True)))
            input = heatmap_crop_origin(heatmap.cpu(), input)

        if args.branch_type == 'fusion':
            test_model1, test_model2 = test_model
            output, heatmap, pool1 = test_model1(torch.autograd.Variable(input.cuda(async = True)))
            inputs = heatmap_crop_origin(heatmap.cpu(), input)
            output, heatmap, pool2 = test_model2(torch.autograd.Variable(inputs.cuda(async = True)))
            pool1 = pool1.view(pool1.size(0), -1)
            pool2 = pool2.view(pool2.size(0), -1)
            input = torch.cat((pool1.cpu(), pool2.cpu()), dim=1)

        input = input.cuda(async = True)
                 
        varInput = torch.autograd.Variable(input, volatile=True)
        varTarget = torch.autograd.Variable(target, volatile=True)    
        varOutput, heatmap, pool = model(varInput)
            
        losstensor = loss(varOutput, varTarget)
        losstensorMean += losstensor
            
        lossVal += losstensor.data[0]
        lossValNorm += 1
            
    outLoss = lossVal / lossValNorm
    losstensorMean = losstensorMean / lossValNorm
        
    return outLoss, losstensorMean

globalmodel = globalnet(args).cuda()
localmodel = localnet(args).cuda()
fusionmodel = fusionnet(args).cuda()

#-------------------- SETTINGS: DATA TRANSFORMS
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
transformList_train = []
transformList_train.append(transforms.Resize(256))
transformList_train.append(transforms.RandomResizedCrop(224))
transformList_train.append(transforms.RandomHorizontalFlip())
transformList_train.append(transforms.ToTensor())
transformList_train.append(normalize)      
transformSequence_train=transforms.Compose(transformList_train)

transformList_test_val = []
transformList_test_val.append(transforms.Resize(256))
transformList_test_val.append(transforms.CenterCrop(224))
transformList_test_val.append(transforms.ToTensor())
transformList_test_val.append(normalize)      
transformSequence_test_val=transforms.Compose(transformList_test_val)

def train():             
    #-------------------- SETTINGS: DATASET BUILDERS
    trBatchSize = 16
    datasetTrain = DatasetGenerator(pathImageDirectory = '../chest-image/images',
                               pathDatasetFile = './dataset/train_1.txt',
                               transform = transformSequence_train)
    datasetVal = DatasetGenerator(pathImageDirectory = '../chest-image/images',
                               pathDatasetFile = './dataset/val_1.txt',
                               transform = transformSequence_test_val)  
    #train_loader = DataLoader(chest_train, batch_size=4, shuffle=True, num_workers=2)
              
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=2, pin_memory=True)
    dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=True)
        
    #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
    optimizer_global = optim.SGD (globalmodel.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    optimizer_local = optim.SGD (localmodel.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    optimizer_fusion = optim.SGD (fusionmodel.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
                
    #-------------------- SETTINGS: LOSS
    loss_global = torch.nn.BCELoss(size_average = True)
    loss_local = torch.nn.BCELoss(size_average = True)
    loss_fusion = torch.nn.BCELoss(size_average = True)
        
    #---- TRAIN THE NETWORK
        
    lossMIN = 100000
    trMaxEpoch = 50
    nnClassCount = 14

    #---- global
    if args.branch_type == 'global':
        print('train for global begin!')
        lr = 0.01
        for epochID in range (0, trMaxEpoch):
            if (epochID+1)%20 == 0:
                lr = adjust_learning_rate(optimizer_global, lr)
                         
            epochTrain (epochID, globalmodel, dataLoaderTrain, optimizer_global, loss_global, None)
            lossVal, losstensor = epochVal (epochID, globalmodel, dataLoaderVal, optimizer_global, loss_global, None)
                       
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save(globalmodel.state_dict(), './model/model_global.pth.tar')
                print ('Global || Epoch [' + str(epochID + 1) + '] [save]  loss= ' + str(lossVal))
            else:
                print ('Global || Epoch [' + str(epochID + 1) + '] [----]  loss= ' + str(lossVal))
        print('train for global finish!')

    #--- local
    if args.branch_type == 'local':
        print('train for local begin!')
        lr = 0.01
    
        test_model = globalnet(args).cuda()
        test_model.load_state_dict(torch.load('./model/model_global.pth.tar'))
        for para in list(test_model.parameters()):
            para.requires_grad=False
        test_model = test_model.cuda()
        localmodel.load_state_dict(torch.load('./model/model_local.pth.tar'))

        for epochID in range (0, trMaxEpoch):
            if (epochID+1)%20 == 0:
                lr = adjust_learning_rate(optimizer_local, lr)

            epochTrain (epochID, localmodel, dataLoaderTrain, optimizer_local, loss_local, test_model)
            lossVal, losstensor = epochVal (epochID, localmodel, dataLoaderVal, optimizer_local, loss_local, test_model)
                       
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save(localmodel.state_dict(), './model/model_local.pth.tar')
                print ('Local || Epoch [' + str(epochID + 1) + '] [save]  loss= ' + str(lossVal))
            else:
                print ('Local || Epoch [' + str(epochID + 1) + '] [----]  loss= ' + str(lossVal))
        print('train for local finish!')

    #---fusion
    if args.branch_type == 'fusion':
        print('train for fusion begin!')
        lr = 0.01
    
        test_model1 = globalnet(args).cuda()
        test_model1.load_state_dict(torch.load('./model/model_global.pth.tar'))
        for para in list(test_model1.parameters()):
            para.requires_grad=False
        test_model2 = localnet(args).cuda()
        test_model2.load_state_dict(torch.load('./model/model_local.pth.tar'))
        for para in list(test_model2.parameters()):
            para.requires_grad=False
        test_model = (test_model1, test_model2)

        for epochID in range (0, trMaxEpoch):
            if (epochID+1)%20 == 0:
                lr = adjust_learning_rate(optimizer_fusion, lr)

            epochTrain (epochID, fusionmodel, dataLoaderTrain, optimizer_fusion, loss_fusion, test_model)
            lossVal, losstensor = epochVal (epochID, fusionmodel, dataLoaderVal, optimizer_fusion, loss_fusion, test_model)
                       
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save(fusionmodel.state_dict(), './model/model_fusion.pth.tar')
                print ('Fusion || Epoch [' + str(epochID + 1) + '] [save]  loss= ' + str(lossVal))
            else:
                print ('Fusion || Epoch [' + str(epochID + 1) + '] [----]  loss= ' + str(lossVal))
        print('train for fusion finish!')

def test():             
    #-------------------- SETTINGS: DATASET BUILDERS
    trBatchSize = 16
    datasetTest = DatasetGenerator(pathImageDirectory = '../chest-image/images',
                               pathDatasetFile = './dataset/test_1.txt',
                               transform = transformSequence_test_val)
              
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=True)
        
    #---- TEST THE NETWORK
        
    nnClassCount = 14

    disease_class = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    #---- global
    if args.branch_type == 'global':
        print('test for global begin!')

        globalmodel.load_state_dict(torch.load('./model/model_global.pth.tar'))
        globalmodel.eval()

        scoreFile = open('./out/'+args.branch_type+'.csv', 'w')
        fileHeader =  ['Image Index'] + disease_class
        writer = csv.writer(scoreFile)
        writer.writerow(fileHeader)

        for batchID, (input, target) in enumerate (dataLoaderTest, 0):
            print('gloabl Testing...'+'||'+'test:'+str(batchID))

            input = input.cuda()

            varInput = torch.autograd.Variable(input)        
            varOutput, heatmap, pool = globalmodel(varInput)

            varOutput = varOutput.data.cpu()
            varOutput = np.ndarray.tolist(varOutput.numpy())
            # print (len(varOutput))
            # print(len(varOutput[0]))
            # print(len(target))
            for j in range(len(target)):
                item = varOutput[j]
                writer.writerow(item)
    
        scoreFile.close()
        
        print('test for global finish!')

    #--- local
    if args.branch_type == 'local':
        print('test for local begin!')

        globalmodel.load_state_dict(torch.load('./model/model_global.pth.tar'))
        globalmodel.eval()
        localmodel.load_state_dict(torch.load('./model/model_local.pth.tar'))
        localmodel.eval()

        scoreFile = open('./out/'+args.branch_type+'.csv', 'w')
        fileHeader =  ['Image Index'] + disease_class
        writer = csv.writer(scoreFile)
        writer.writerow(fileHeader)

        for batchID, (input, target) in enumerate (dataLoaderTest):
            print('local Testing...'+'||'+'test:'+str(batchID))

            varInput = torch.autograd.Variable(input.cuda())    
            varOutput, heatmap, pool = globalmodel(varInput)
            inputs = heatmap_crop_origin(heatmap.cpu(), input)

            varOutput, heatmap, pool = localmodel(torch.autograd.Variable(inputs.cuda()))

            varOutput = varOutput.data.cpu()
            varOutput = np.ndarray.tolist(varOutput.numpy())

            for j in range(len(target)):
                item = varOutput[j]
                writer.writerow(item)
    
        scoreFile.close()

        print('test for local finish!')

    #---fusion
    if args.branch_type == 'fusion':
        print('test for fusion begin!')
        
        globalmodel.load_state_dict(torch.load('./model/model_global.pth.tar'))
        globalmodel.eval()
        localmodel.load_state_dict(torch.load('./model/model_local.pth.tar'))
        localmodel.eval()
        fusionmodel.load_state_dict(torch.load('./model/model_fusion.pth.tar'))
        fusionmodel.eval()

        scoreFile = open('./out/'+args.branch_type+'.csv', 'w')
        fileHeader =  ['Image Index'] + disease_class
        writer = csv.writer(scoreFile)
        writer.writerow(fileHeader)

        for batchID, (input, target) in enumerate (dataLoaderTest):
            print('fusion Testing...'+'||'+'test:'+str(batchID))

            varInput = torch.autograd.Variable(input.cuda())    
            varOutput, heatmap, pool1 = globalmodel(varInput)
            inputs = heatmap_crop_origin(heatmap.cpu(), input)
            varOutput, heatmap, pool2 = localmodel(torch.autograd.Variable(inputs.cuda()))

            pool1 = pool1.view(pool1.size(0), -1)
            pool2 = pool2.view(pool2.size(0), -1)
            inputss = torch.cat((pool1.cpu(), pool2.cpu()), dim=1)
            varOutput, heatmap, pool = fusionmodel(torch.autograd.Variable(inputss.cuda()))

            varOutput = varOutput.data.cpu()
            varOutput = np.ndarray.tolist(varOutput.numpy())

            for j in range(len(target)):
                item = varOutput[j]
                writer.writerow(item)
    
        scoreFile.close()

        print('test for fusion finish!')

if __name__ == '__main__':
    if args.use == 'train':
        train()
    if args.use == 'test':
        test()