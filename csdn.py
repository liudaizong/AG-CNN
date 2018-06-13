import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import roc_curve

df1 = pd.read_csv('./out/local.csv')
pred = df1.values[:,0:]
# df2 = pd.read_csv('test_l.csv')
# gt = df2.values[:,1:].astype('float')
fileDescriptor = open('./dataset/test_1.txt', "r")
fr = fileDescriptor.readlines()
gt = np.array([i.split('\n')[0].split(' ')[1:] for i in fr]).astype('float')
CLASS_NAMES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
style = ['r-','g-','b-','y-','k-','c-','m-','r--','g--','b--','y--','k--','c--','m--']
print pred.shape
print gt.shape
print float(gt[38,0])
average_roc = 0.0
for i in range(14):
	roc_value = roc_auc_score(gt[:,i], pred[:,i],sample_weight=None)
	print CLASS_NAMES[i], ':', roc_value 
	average_roc += roc_value
	fpr, tpr, thresholds = roc_curve(gt[:,i], pred[:,i])
	plt.plot(fpr, tpr,style[i],label=CLASS_NAMES[i])
print 'average_roc: ', average_roc/14
plt.title('roc')
plt.xlabel('FPR_array')
plt.ylabel('TPR_array')
plt.legend()
plt.show()