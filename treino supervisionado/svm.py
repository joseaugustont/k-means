import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from skimage import feature
import seaborn as sns
from sklearn import metrics, svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
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

def getDados(pasta):
    types = ('*.tif', '*jpg2')
    diretory = 'C:/Users/augus/Documents/ProjPIBIC/k-means/'
    fill = ['0/', '1/', '2/', '3/', '4/']
    images = []
    y_true = []
    imagePaths = []
    inicio = 0

    def addImages(imPaths, inicio, images):
        for i in imPaths:
            images.append(cv2.imread('{0}'.format(i),-1))
            imagePaths.append(i)
            y_true.append(inicio)
        return images

    for f in fill:
        imagePaths2 = []
        for files in types :
            imagePaths2.extend(sorted(glob.glob(diretory + pasta  + "/" + f + files)))
        images = addImages(imagePaths2, inicio, images)
        inicio +=1

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

    whitened = [[] for i in range(0,len(imagePaths))]

    for i in range(len(whitened)):
        a = np.concatenate((whitened_prepare[i][0], whitened_prepare[i][1]), axis=0)
        b = np.concatenate((whitened_prepare[i][2], whitened_prepare[i][3]), axis=0)
        whitened[i] = np.concatenate((a,b), axis=0)

    whitened = np.array(whitened)
    return whitened, y_true


x_train, y_train = getDados("train")
sns.countplot(y_train)
plt.show()

x_test, y_test = getDados("test")
sns.countplot(y_test)
plt.show()

clf = svm.SVC(kernel='linear', probability=True)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
y_predict = [i for i in y_predict]

print("Acurácia - {0}".format(accuracy_score(y_test, y_predict)))
print("f1 score: ponderada - {0}".format(f1_score(y_test, y_predict, average='weighted')))
print("Recall score: ponderada - {0}".format(recall_score(y_test, y_predict, average='weighted', zero_division=0)))
print("Precision score: ponderada - {0}".format(precision_score(y_test, y_predict, average='weighted', zero_division=0)))
