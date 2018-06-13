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

parser.add_argument('--model', type=str, default='resnet50',
                    help='basic model (resneat50 or densenet121)')

args = parser.parse_args()

print(args)

df = pd.read_csv('./dataset/BBox_List_2017.csv')
# fr = open('./dataset/bbox.txt','w')
# print(df.values.shape[0])
# for i in range(df.values.shape[0]):
#     img_name = df.values[i,0]
#     fr.write('image_003/'+img_name+' 1 0 3 1 0 3 1 0 3 1 0 3\n')
# fileDescriptor = open('./dataset/bbox.txt', "r")
        
# line = True
# i = 0
# while line:
#     i += 1
#     line = fileDescriptor.readline()
#     # print (line)
            
#     if line:
          
#         lineItems = line.split()
                
#         print(lineItems[0].split('/')[-1])
#     	print(i)

#-------------------- SETTINGS: DATASET BUILDERS
trBatchSize = 1
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList_test_val = []
transformList_test_val.append(transforms.Resize(256))
transformList_test_val.append(transforms.CenterCrop(224))
transformList_test_val.append(transforms.ToTensor())
transformList_test_val.append(normalize)      
transformSequence_test_val=transforms.Compose(transformList_test_val)
datasetTest = DatasetGenerator(pathImageDirectory = '../chest-image/images',
                               pathDatasetFile = './dataset/bbox.txt',
                               transform = transformSequence_test_val)
              
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, shuffle=False, num_workers=2, pin_memory=True)
        
#---- TEST THE NETWORK
        
nnClassCount = 14

disease_class = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

#---- global
print('test for global begin!')

globalmodel = globalnet(args).cuda()
globalmodel.load_state_dict(torch.load('./model/model_global.pth.tar'))
globalmodel.eval()

def count_iou(thre):
    print('test thre: %f'%thre)
    iou_gt = 0.1
    iou = 0.0
    total_num = df.values.shape[0]
    num = 0
    for batchID, (input, target) in enumerate (dataLoaderTest, 0):
        # print('gloabl Testing...'+'||'+'test:'+str(batchID))
        input = input.cuda()

        varInput = torch.autograd.Variable(input)        
        varOutput, heatmap, pool = globalmodel(varInput)
        heatmap = heatmap.cpu()
        # print (heatmap.shape)
        batchsize = heatmap.size(0)
        # print (batchID)
        for batch in range(batchsize):
            heatmap_one = torch.abs(heatmap[batch])
            heatmap_two = torch.max(heatmap_one, dim=0)[0].squeeze(0)
            max1 = torch.max(heatmap_two)
            min1 = torch.min(heatmap_two)
            heatmap_two = (heatmap_two - min1) / (max1 - min1)
            # max_value = torch.max(heatmap_two)
            # min_value = torch.min(heatmap_two)
            # heatmap_two = (heatmap_two - min_value) / (max_value - min_value)
            # print (heatmap_two)
            heatmap_two[heatmap_two > thre] = 1
            heatmap_two[heatmap_two != 1] = 0
            where = torch.from_numpy(np.argwhere(heatmap_two.numpy() == 1))
            # print (where)
            # print (torch.min(where, dim =0)[0][0])
            # print (torch.max(where, dim=0)[0][0])
            # print (torch.min(where, dim =0)[0][1])
            # print (torch.max(where, dim =0)[0][1])
            xmin = int((torch.min(where, dim =0)[0][0])*224/7)
            xmax = int(torch.max(where, dim=0)[0][0]*224/7)
            ymin = int(torch.min(where, dim =0)[0][1]*224/7)
            ymax = int(torch.max(where, dim =0)[0][1]*224/7)
            # print(xmin,xmax,ymin,ymax)
            if xmin == xmax:
                xmin = int((torch.min(where, dim =0)[0][0])*224/7)
                xmax = int((torch.max(where, dim=0)[0][0] + 1)*224/7)
            if ymin == ymax:
                ymin = int((torch.min(where, dim =0)[0][1])*224/7)
                ymax = int((torch.max(where, dim =0)[0][1] + 1)*224/7)
            # print (xmin, xmax, ymin, ymax)
            carea = (xmax - xmin) * (ymax - ymin)

            # print (df.values[batchID, 2:])
            [x, y, w, h] = df.values[batchID, 2:6]
            xmin_gt = int(y*224/1024)
            xmax_gt = int((y+h)*224/1024)
            ymin_gt = int(x*224/1024)
            ymax_gt = int((x+w)*224/1024)
            # print(xmin_gt,xmax_gt,ymin_gt,ymax_gt)
            garea = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)

            x1 = max(xmin, xmin_gt)
            y1 = max(ymin, ymin_gt)
            x2 = min(xmax, xmax_gt)
            y2 = min(ymax, ymax_gt)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = w * h 
 
            iou_old = float(area) / (carea + garea - area)
            # if area == carea or area == garea:
            #     iou_old = 1.0
            iou += iou_old
            # if area >= iou_gt*carea or area >= iou_gt*garea:
            #     num += 1
            # print(iou)
            # print(carea)
            # print(garea)
            # print(area)
        # break
    # print(float(num)/total_num)
    iou_average = iou/total_num
    return iou_average

if __name__ == '__main__':
    scoreFile = open('./out/iou.csv', 'w')
    fileHeader =  ['thre'] + ['iou_average']
    writer = csv.writer(scoreFile)
    writer.writerow(fileHeader)
    for thre in np.arange(0, 1, 0.1):
        iou_average = count_iou(thre)
        item = [thre, iou_average]
        writer.writerow(item)
    
    scoreFile.close()