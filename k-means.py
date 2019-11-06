import numpy as np
from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import scipy.stats as sc
import matplotlib.pyplot as plt
from skimage import feature
import pandas as pd
import shutil
import cv2
import glob
import math
import os

def LBP(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = feature.local_binary_pattern(gray, 10, 5, method="default")
    return features

types = ('*.tif', '*jpg2')
diretory = 'C:/Users/augus/Documents/ProjPIBIC/k-means/images/'
imagePaths = []
for files in types :
	imagePaths.extend(sorted(glob.glob(diretory + files)))

images = [cv2.imread('{0}'.format(i),-1) for i in imagePaths]

structure = {'image_name':[], 'histo-blue':[],'histo-green':[], 'histo-red':[], 'histo-lbp':[]}
df = pd.DataFrame(data=structure)

for i in range(len(images)):
  b = cv2.calcHist([images[i]],[0],None,[256],[0,256])
  g = cv2.calcHist([images[i]],[1],None,[256],[0,256])
  r = cv2.calcHist([images[i]],[2],None,[256],[0,256])
  image = LBP(images[i])
  lbp,bins = np.histogram(image.ravel(),256,[0,256])
  structure = {'image_name':imagePaths[i], 'histo-blue':b.ravel(),'histo-green':g.ravel(), 'histo-red':r.ravel(), 'histo-lbp':lbp}
  df = df.append(structure, ignore_index=True) 

df2 = df.drop('image_name', 1)
whitened_prepare = df2.values

for i in range(len(whitened_prepare)):
  for j in range(len(whitened_prepare[i])):
    whitened_prepare[i][j] = np.float32(whitened_prepare[i][j])

whitened = [[] for i in range(54)]

for i in range(len(whitened)):
  a = np.concatenate((whitened_prepare[i][0], whitened_prepare[i][1]), axis=0)
  b = np.concatenate((whitened_prepare[i][2], whitened_prepare[i][3]), axis=0)
  whitened[i] = np.concatenate((a,b), axis=0)

whitened = whiten(whitened)
clusters= 5
centroide,label = kmeans2(data=whitened,k=clusters,iter=3000)

saida = pd.DataFrame(data=[], columns=['image_name', 'label'])

saida['label'] = label
saida['image_name'] = imagePaths

saida1 = []
for i in range(0,clusters):
  saida1.append(saida[saida['label']==i].values)

classe = 'classe-'
diretory = 'C:/Users/augus/Documents/ProjPIBIC/k-means/results/'

for i in range(0,clusters):
    os.mkdir(diretory + classe + str(i))

for i in range(0,clusters):
  for j in range(len(saida1[i])):
    shutil.copy(str(saida1[i][j][0]), diretory + classe + str(saida1[i][j][1]) + '/')
