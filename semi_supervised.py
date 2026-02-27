import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler

DATASET="asl_alphabet_train/asl_alphabet_train"

features=[]
labels=[]
label_names=os.listdir(DATASET)

print("Loading data...")

for i,label in enumerate(tqdm(label_names)):
    folder=os.path.join(DATASET,label)

    for imgname in os.listdir(folder)[:200]:
        path=os.path.join(folder,imgname)

        img=cv2.imread(path)
        img=cv2.resize(img,(32,32))

        features.append(img.flatten())
        labels.append(i)

X=np.array(features)
y=np.array(labels)

# normalize
X=StandardScaler().fit_transform(X)

# hide 90% labels
rng=np.random.RandomState(42)
mask=rng.rand(len(y)) < 0.1
y_partial=np.copy(y)
y_partial[~mask]=-1

print("Training semi-supervised model...")

model=LabelSpreading()
model.fit(X,y_partial)

pred=model.transduction_

accuracy=(pred==y).mean()

print("Accuracy using only 10% labels:",accuracy)
