
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Path to dataset
DATASET="asl_alphabet_train/asl_alphabet_train"

# Pick one class as "positive" (e.g., letter A)
POSITIVE_CLASS="A"

features=[]
labels=[]

print("Extracting features...")

for label in os.listdir(DATASET):
    folder=os.path.join(DATASET,label)

    for imgname in os.listdir(folder)[:150]:  # limit for speed
        path=os.path.join(folder,imgname)
        img=cv2.imread(path)
        img=cv2.resize(img,(32,32))
        features.append(img.flatten())
        if label == POSITIVE_CLASS:
            labels.append(1)  # positive
        else:
            labels.append(0)  # unlabeled/negative

X=np.array(features)
y=np.array(labels)

# normalize
X=StandardScaler().fit_transform(X)

# PU Learning using One-Class SVM
X_positive = X[y==1]

print("Training One-Class SVM on positive class only...")
model = OneClassSVM(gamma='auto', nu=0.1)
model.fit(X_positive)

# Predict on all data
pred = model.predict(X)

# convert prediction to 0/1
# +1 = predicted positive, -1 = predicted negative
pred_binary = (pred==1).astype(int)

accuracy = (pred_binary == y).mean()

print("PU model accuracy:", accuracy)

