import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATASET="asl_alphabet_train/asl_alphabet_train"

features=[]
labels=[]

print("Extracting features...")

for label in tqdm(os.listdir(DATASET)):
    folder=os.path.join(DATASET,label)

    for imgname in os.listdir(folder)[:300]:  # limit for speed
        path=os.path.join(folder,imgname)

        img=cv2.imread(path)
        img=cv2.resize(img,(64,64))

        features.append(img.flatten())
        labels.append(label)

X=np.array(features)

# normalize
scaler=StandardScaler()
X=scaler.fit_transform(X)

# PCA reduce
pca=PCA(n_components=2)
X2=pca.fit_transform(X)

# clustering
kmeans=KMeans(n_clusters=29)
clusters=kmeans.fit_predict(X)

# plot
plt.figure(figsize=(8,6))
plt.scatter(X2[:,0],X2[:,1],c=clusters,cmap="tab20")
plt.title("Unsupervised Clustering of Signs")
plt.savefig("outputs/unsupervised_clusters.png")
plt.show()

print("Saved plot → outputs/unsupervised_clusters.png")
