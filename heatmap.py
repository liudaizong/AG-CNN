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
from PIL import Image

from DatasetGenerator import DatasetGenerator 
from model import *

parser = argparse.ArgumentParser(description='AG-CNN')

parser.add_argument('--model', type=str, default='resnet50',
                    help='basic model (resneat50 or densenet121)')

args = parser.parse_args()

print(args)

def heatmap_crop(heatmap):
	batchsize = heatmap.size(0)
	for batch in range(batchsize):
		heatmap_one = torch.abs(heatmap[batch])
		heatmap_two = torch.max(heatmap_one, dim=0)[0].squeeze(0)
		# max1 = torch.max(heatmap_two)
		# min1 = torch.min(heatmap_two)
		# heatmap_two = (heatmap_two - min1) / (max1 - min1)
	return heatmap_two

def heatmap_crop_origin(heatmap, origin_img):
	batchsize = heatmap.size(0)
	thre = 0.6
	img = torch.randn(batchsize, 3, 224, 224)
	for batch in range(batchsize):
		heatmap_one = torch.abs(heatmap[batch])
		# heatmap_two = torch.mean(heatmap_one, dim=0).squeeze(0)
		heatmap_two = torch.max(heatmap_one, dim=0)[0].squeeze(0)
		max1 = torch.max(heatmap_two)
		min1 = torch.min(heatmap_two)
		heatmap_two = (heatmap_two - min1) / (max1 - min1)
		# print(heatmap_two)
		# print (heatmap_two)
		heatmap_two[heatmap_two > thre] = 1
		heatmap_two[heatmap_two != 1] = 0
		print(heatmap_two)
		where = torch.from_numpy(np.argwhere(heatmap_two.numpy() == 1))
		# print (where)
		# print (torch.min(where, dim =0)[0][0])
		# print (torch.max(where, dim=0)[0][0])
		# print (torch.min(where, dim =0)[0][1])
		# print (torch.max(where, dim =0)[0][1])
		xmin = int((torch.min(where, dim =0)[0][0])*1024/7)
		xmax = int(torch.max(where, dim=0)[0][0]*1024/7)
		ymin = int(torch.min(where, dim =0)[0][1]*1024/7)
		ymax = int(torch.max(where, dim =0)[0][1]*1024/7)
		print(xmin, xmax, ymin, ymax)
		if xmin == xmax:
			xmin = int((torch.min(where, dim =0)[0][0])*1024/7)
			xmax = int((torch.max(where, dim=0)[0][0] + 1)*1024/7)
		if ymin == ymax:
			ymin = int((torch.min(where, dim =0)[0][1])*1024/7)
			ymax = int((torch.max(where, dim =0)[0][1] + 1)*1024/7)
		sliced = transforms.ToPILImage()(origin_img[:, xmin:xmax, ymin:ymax])
		img_one = sliced.resize((224, 224), Image.ANTIALIAS)
		# print(np.transpose(img_one, (2, 1, 0)).shape)
		img[batch] = transforms.ToTensor()(img_one)
		# sliced = ?origin_img
	return img, (xmin,xmax,ymin,ymax)

def bbox(pathImageFile, name):
	df1 = pd.read_csv('./dataset/BBox_List_2017.csv')
	imgname = df1.values[:,0]
	for i in range(len(imgname)):
		if imgname[i] in pathImageFile:
			rectangle = df1.values[i,:]

	red = (0, 0, 255)
	img = cv2.imread(pathImageFile)
	img = cv2.resize(img, (224, 224))
	origin = (int(rectangle[2]*224/1024), int(rectangle[3]*224/1024))
	end = (int((rectangle[2]+rectangle[4])*224/1024), int((rectangle[3]+rectangle[5])*224/1024))
	cv2.rectangle(img, origin, end, red, 3)
	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	cv2.imwrite('./heatmap/'+name+'/img.jpg', img)

def boundry(pathImageFile, name, xy):
	red = (0, 0, 255)
	img = cv2.imread(pathImageFile)
	img = cv2.resize(img, (224, 224))
	(xmin,xmax,ymin,ymax) = xy
	xmin = int(xmin*224/1024)
	xmax = int(xmax*224/1024)
	ymin = int(ymin*224/1024)
	ymax = int(ymax*224/1024)
	cv2.rectangle(img, (ymin, xmin), (ymax, xmax), red, 3)
	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	cv2.imwrite('./heatmap/'+name+'/boundry.jpg', img)

def compare(pathImageFile, name, xy):
	df1 = pd.read_csv('./dataset/BBox_List_2017.csv')
	imgname = df1.values[:,0]
	for i in range(len(imgname)):
		if imgname[i] in pathImageFile:
			rectangle = df1.values[i,:]

	red = (0, 0, 255)
	img = cv2.imread(pathImageFile)
	img = cv2.resize(img, (224, 224))
	origin = (int(rectangle[2]*224/1024), int(rectangle[3]*224/1024))
	end = (int((rectangle[2]+rectangle[4])*224/1024), int((rectangle[3]+rectangle[5])*224/1024))
	cv2.rectangle(img, origin, end, red, 3)

	blue = (255, 0, 0)
	(xmin,xmax,ymin,ymax) = xy
	xmin = int(xmin*224/1024)
	xmax = int(xmax*224/1024)
	ymin = int(ymin*224/1024)
	ymax = int(ymax*224/1024)
	cv2.rectangle(img, (ymin, xmin), (ymax, xmax), blue, 3)
	# cv2.imshow('img', img)
	# cv2.waitKey(0)
	cv2.imwrite('./heatmap/'+name+'/compare.jpg', img)

def generate(globalmodel, localmodel, pathImageFile, transformSequence_test_val, name):
        
    globalmodel.eval()
    localmodel.eval()
        
    imageData = Image.open(pathImageFile).convert('RGB')
    imageData = transformSequence_test_val(imageData)
    imageData = imageData.unsqueeze_(0)

    input = torch.autograd.Variable(imageData)

    output1, heatmap1, pool1 = globalmodel(input.cuda())

    npHeatmap = heatmap_crop(heatmap1.cpu()).numpy()
    imgOriginal = cv2.imread(pathImageFile, 1)
    imgOriginal = cv2.resize(imgOriginal, (224, 224))
    # cv2.imshow('imgOriginal', imgOriginal)
    # cv2.waitKey(0)
    cv2.imwrite('./heatmap/'+name+'/imgOriginal.jpg', imgOriginal)

    bbox(pathImageFile, name)
        
    cam = npHeatmap / np.max(npHeatmap)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    # cv2.imshow('heatmap', heatmap)
    # cv2.waitKey(0)
    cv2.imwrite('./heatmap/'+name+'/heatmap.jpg', heatmap)
              
    img1 = heatmap * 0.3 + imgOriginal * 0.5
    # cv2.imshow('img1', img1)
    # cv2.waitKey(0)
    cv2.imwrite('./heatmap/'+name+'/img1.jpg', img1)

    inputs, xy = heatmap_crop_origin(heatmap1.cpu(), transforms.ToTensor()(Image.open(pathImageFile).convert('RGB')))
    boundry(pathImageFile, name, xy)
    compare(pathImageFile, name, xy)
    # inputs = heatmap_crop_origin(heatmap1.cpu(), imageData.squeeze(0))
    # print(inputs.shape)
    # cv2.imshow('inputs', inputs[0, :, :, :].numpy())
    # cv2.waitKey(0)
    # cv2.imwrite('./heatmap/global/inputs.jpg', inputs[0, :, :, :].numpy())
    sliced = transforms.ToPILImage()(inputs[0,:,:,:])
    sliced.save('./heatmap/'+name+'/sliced.jpg')

    output2, heatmap2, pool2 = localmodel(torch.autograd.Variable(inputs.cuda()))

    npHeatmap = heatmap_crop(heatmap2.cpu()).numpy()
    imgOriginal = cv2.imread('./heatmap/'+name+'/sliced.jpg')
        
    cam = npHeatmap / np.max(npHeatmap)
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
    img2 = heatmap * 0.3 + imgOriginal * 0.5
    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)
    cv2.imwrite('./heatmap/'+name+'/img2.jpg', img2)
        

if __name__ == '__main__':
	if not os.path.exists('./heatmap'):
		os.mkdir('./heatmap')
	if not os.path.exists('./heatmap/global'):
		os.mkdir('./heatmap/global')


	globalmodel = globalnet(args).cuda()
	globalmodel.load_state_dict(torch.load('./model/model_global.pth.tar'))
	localmodel = localnet(args).cuda()
	localmodel.load_state_dict(torch.load('./model/model_local.pth.tar'))

	normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

	transformList_test_val = []
	transformList_test_val.append(transforms.Resize(256))
	transformList_test_val.append(transforms.CenterCrop(224))
	transformList_test_val.append(transforms.ToTensor())
	transformList_test_val.append(normalize)      
	transformSequence_test_val=transforms.Compose(transformList_test_val)

	fr = pd.read_csv('./dataset/BBox_List_2017.csv')
	img_list = fr.values[:,0]
	for i in range(len(img_list)):
		pathImageFile = '../chest-image/images'+'/'+img_list[i]
		if not os.path.exists('./heatmap/'+img_list[i]):
			os.mkdir('./heatmap/'+img_list[i])
		generate(globalmodel, localmodel, pathImageFile, transformSequence_test_val,img_list[i])