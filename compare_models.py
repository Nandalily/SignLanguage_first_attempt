import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATASET="asl_alphabet_train/asl_alphabet_train"

features=[]
labels=[]
class_names=os.listdir(DATASET)

print("Extracting features...")

for idx,label in enumerate(class_names):
    folder=os.path.join(DATASET,label)

    for i,imgname in enumerate(os.listdir(folder)[:120]):
        path=os.path.join(folder,imgname)
        img=cv2.imread(path)
        img=cv2.resize(img,(32,32))
        features.append(img.flatten())
        labels.append(idx)

X=np.array(features)
y=np.array(labels)

# Normalize
X=StandardScaler().fit_transform(X)

results={}

# -----------------------
# 1. KMEANS
# -----------------------
print("Running KMeans...")
kmeans=KMeans(n_clusters=len(class_names),n_init=10)
clusters=kmeans.fit_predict(X)

# map clusters → labels
mapping={}
for i in range(len(class_names)):
    mask=(clusters==i)
    if np.sum(mask)==0:
        continue
    mapping[i]=np.bincount(y[mask]).argmax()

mapped=[mapping[c] for c in clusters]
results["KMeans"]=accuracy_score(y,mapped)

# -----------------------
# 2. LABEL SPREADING
# -----------------------
print("Running Label Spreading...")
y_partial=y.copy()

# hide most labels
rng=np.random.RandomState(42)
mask=rng.rand(len(y))<0.9
y_partial[mask]=-1

lp=LabelSpreading(kernel='knn',alpha=0.2)
lp.fit(X,y_partial)

pred_lp=lp.transduction_
results["LabelSpreading"]=accuracy_score(y,pred_lp)

# -----------------------
# 3. ONE CLASS SVM
# -----------------------
print("Running One-Class SVM...")
svm=OneClassSVM(gamma='auto').fit(X[y==0])
pred=svm.predict(X)

# convert output
pred_labels=(pred==1).astype(int)
true_labels=(y==0).astype(int)

results["OneClassSVM"]=accuracy_score(true_labels,pred_labels)

# -----------------------
# RESULTS TABLE
# -----------------------
print("\nMODEL COMPARISON RESULTS")
print("-"*35)

for k,v in results.items():
    print(f"{k:<20} {v:.4f}")

best=max(results,key=results.get)

print("\nBest Model →",best)
