import torch.optim as optim
import torch
import numpy as np
from PIL import Image
import csv
import pandas as pd 
from torchvision import transforms

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/10
    return lr/10

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]
    return z

def heatmap_crop_origin(heatmap, origin_img):
	batchsize = heatmap.size(0)
	thre = 0.7
	img = torch.randn(batchsize, 3, 224, 224)
	for batch in range(batchsize):
		heatmap_one = torch.abs(heatmap[batch])
		heatmap_two = torch.max(heatmap_one, dim=0)[0].squeeze(0)
		max1 = torch.max(heatmap_two)
		min1 = torch.min(heatmap_two)
		heatmap_two = (heatmap_two - min1) / (max1 - min1)
		# print (heatmap_two)
		heatmap_two[heatmap_two > thre] = 1
		heatmap_two[heatmap_two != 1] = 0
		# print(heatmap_two)
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
		if xmin == xmax:
			xmin = int((torch.min(where, dim =0)[0][0])*224/7)
			xmax = int((torch.max(where, dim=0)[0][0] + 1)*224/7)
		if ymin == ymax:
			ymin = int((torch.min(where, dim =0)[0][1])*224/7)
			ymax = int((torch.max(where, dim =0)[0][1] + 1)*224/7)
		sliced = transforms.ToPILImage()(origin_img[batch][:, xmin:xmax, ymin:ymax])
		img_one = sliced.resize((224, 224), Image.ANTIALIAS)
		# print(np.transpose(img_one, (2, 1, 0)).shape)
		img[batch] = transforms.ToTensor()(img_one)

	return img

def save_to_csv(output, branch_type, labels):
	disease_class = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
	scoreFile = open('./out/'+branch_type+'.csv', 'w')
	fileHeader =  ['Image Index'] + disease_class
	writer = csv.writer(scoreFile)
	writer.writerow(fileHeader)

	for i in output:
		print ('saving... '+str(output.index(i)))
		for j in range(len(labels)):
			item = i[j]
			writer.writerow(item)
	
	scoreFile.close()

if __name__ == '__main__':
	heat = torch.Tensor([[1,1,0,0],
					[1,1,0,0],
					[1,1,0,0],
					[1,1,0,0]])
	heat = heat.numpy()
	# heat[heat == 1] = 2
	# heat = heat.numpy()
	# where = torch.from_numpy(np.argwhere(heat == 2))
	# print where
	# xmmin = torch.min(where, dim =0)[0]
	# xmax = torch.max(where, dim=0)[0]
	# print xmmin
	# print xmax
	# a =  torch.randn(7,7,3).numpy()
	# # print a
	# print a
	# print np.transpose(a, (2,1,0))
	a = torch.ones(1,7)
	print torch.cat((a,a), dim=1)
	
