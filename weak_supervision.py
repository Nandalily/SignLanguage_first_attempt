import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

DATASET="asl_alphabet_train/asl_alphabet_train"

features=[]
labels=[]
class_names=os.listdir(DATASET)

print("Extracting features...")

for idx,label in enumerate(class_names):
    folder=os.path.join(DATASET,label)

    for i,imgname in enumerate(os.listdir(folder)[:120]):  # limit
        path=os.path.join(folder,imgname)
        img=cv2.imread(path)
        img=cv2.resize(img,(32,32))
        features.append(img.flatten())

        # Only label first 10 images per class
        if i < 10:
            labels.append(idx)
        else:
            labels.append(-1)  # unlabeled

X=np.array(features)
y=np.array(labels)

# normalize
X=StandardScaler().fit_transform(X)

print("Training Label Spreading model...")
model=LabelSpreading(kernel='knn',alpha=0.2)
model.fit(X,y)

pred=model.transduction_

print("\nClassification Report (on labeled data):")
mask=y!=-1
print(classification_report(y[mask],pred[mask]))
