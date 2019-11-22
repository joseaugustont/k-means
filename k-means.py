import numpy as np
from numpy import array
from scipy.cluster.vq import vq, whiten, kmeans2
import scipy.stats as sc
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn import metrics
import pandas as pd
import shutil
import cv2
import glob
import math
import os

def LBP(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = feature.local_binary_pattern(gray, 8, 1, method="uniform")
    return features

types = ('*.tif', '*jpg2')
diretory = 'C:/Users/augus/Documents/ProjPIBIC/k-means/zeis/'
imagePaths = []
for files in types :
	imagePaths.extend(sorted(glob.glob(diretory + files)))

images = [cv2.imread('{0}'.format(i),-1) for i in imagePaths]
y_true = [0 for i in imagePaths]

diretory = 'C:/Users/augus/Documents/ProjPIBIC/k-means/nao-zeis/'
cont = len(imagePaths)
imagePaths2 = []
for files in types:
  imagePaths2.extend(sorted(glob.glob(diretory + files)))
for i in imagePaths2:
  imagePaths.append(i)
  y_true.append(1)
  images.append(cv2.imread('{0}'.format(i),-1))

structure = {'image_name':[], 'histo-blue':[],'histo-green':[], 'histo-red':[], 'histo-lbp':[]}
df = pd.DataFrame(data=structure)

for i in range(len(images)):
  b = cv2.calcHist([images[i]],[0],None,[64],[0,256])
  g = cv2.calcHist([images[i]],[1],None,[64],[0,256])
  r = cv2.calcHist([images[i]],[2],None,[64],[0,256])
  image = LBP(images[i])
  lbp,bins = np.histogram(image.ravel(),10)
  structure = {'image_name':imagePaths[i], 'histo-blue':b.ravel(),'histo-green':g.ravel(), 'histo-red':r.ravel(), 'histo-lbp':lbp}
  df = df.append(structure, ignore_index=True) 

df2 = df.drop('image_name', 1)
whitened_prepare = df2.values

for i in range(len(whitened_prepare)):
  for j in range(len(whitened_prepare[i])):
    whitened_prepare[i][j] = np.float32(whitened_prepare[i][j])

whitened = [[] for i in range(0,len(imagePaths))]

for i in range(len(whitened)):
  a = np.concatenate((whitened_prepare[i][0], whitened_prepare[i][1]), axis=0)
  b = np.concatenate((whitened_prepare[i][2], whitened_prepare[i][3]), axis=0)
  whitened[i] = np.concatenate((a,b), axis=0)

#whitened = whiten(whitened)
whitened = np.array(whitened)
#clusters = 4
#centroide,label = kmeans2(data=whitened,k=clusters, iter=30000)
inertias=[]
K = range(1,11)
for i in K:
    kmeans = KMeans(n_clusters=i, max_iter=1000).fit(whitened)
    inertias.append(kmeans.inertia_)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.savefig('cotovelo-new2.jpg')
plt.show()

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

print(optimal_number_of_clusters(inertias))

clusters = int(input())

kmeans = KMeans(n_clusters=clusters, max_iter=1000).fit(whitened)
label = kmeans.labels_

print("Inercia - {0}. Silhueta - {1}".format(kmeans.inertia_, silhouette_score(whitened, label, metric='euclidean')))
print("Acurácia - {0}".format(accuracy_score(y_true, label)))
print("f1 score: ponderada - {0}, micro - {1}".format(f1_score(y_true, label, average='weighted'), f1_score(y_true, label, average='micro')))
print("Recall score: ponderada - {0}, micro - {1}".format(recall_score(y_true, label, average='weighted'), recall_score(y_true, label, average='micro')))
print("Precision score: ponderada - {0}, micro - {1}".format(precision_score(y_true, label, average='weighted'), precision_score(y_true, label, average='micro')))

saida = pd.DataFrame(data=[], columns=['image_name', 'label'])

saida['image_name'] = imagePaths
saida['label'] = label

saida1 = []
for i in range(0,clusters):
  saida1.append(saida[saida['label']==i].values)

r = [0 for i in saida1]
s = [0 for i in saida1]
for i in range(len(saida1)):
  for j in range(len(saida1[i])):
    if "n-zeis" in saida1[i][j][0]:
      r[i] +=1
    if not("n-zeis" in saida1[i][j][0]):
      s[i] +=1

print (r, s)

classe = 'classe-'
diretory = 'C:/Users/augus/Documents/ProjPIBIC/k-means/results/'

for i in range(0,clusters):
    os.mkdir(diretory + classe + str(i))

for i in range(0,clusters):
  for j in range(len(saida1[i])):
    shutil.copy(str(saida1[i][j][0]), diretory + classe + str(saida1[i][j][1]) + '/')
