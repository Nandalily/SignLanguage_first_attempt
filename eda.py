import os    #ets pyhon interact with folders and files
import cv2   #cv library for reading images
import numpy as np    #used for numerical operations, arrays, matrices,math. statistics etc...
import pandas as pd   #storing data in tables
import matplotlib.pyplot as plt  #for ploting graphs and visualization
import seaborn as sns #''tion''ploting & visualiza
from tqdm import tqdm   #adds a progress bar when looping through files
from sklearn.decomposition import PCA    #tool for dimension reduction
from sklearn.cluster import KMeans  #for clustering
from sklearn.manifold import TSNE   #advanced visualization

DATASET_PATH = "asl_alphabet_train/asl_alphabet_train"

classes = os.listdir(DATASET_PATH)    #reads all folder names

data = []  #empty list to store extracted data

print("Scanning dataset...")    #progress message

for label in tqdm(classes):    #loop through each class folder
    folder = os.path.join(DATASET_PATH, label)  #creates path to the folder
    for imgname in os.listdir(folder):   #loops through every image inside the class
        path = os.path.join(folder, imgname)    #full path to an image

        img = cv2.imread(path)    #loads image in an array
        if img is None:      #if broken or unreadable, skip
            continue

        h,w,_ = img.shape    #extracts height and width and channels, ignores _
        mean = img.mean()  #average imagge brightness
        std = img.std()     #pixel intensity variation

        data.append([label,h,w,mean,std])    #store all info. about image

df = pd.DataFrame(data,columns=["label","height","width","mean","std"])  #creates structured data tables

os.makedirs("outputs",exist_ok=True)   #creates the output folder if it doesnt exist


#VISUALIZATION
print("Generating plots...")  #status message

# 1 Class distribution
plt.figure()
df["label"].value_counts().plot(kind="bar")  #counts images per class
plt.title("Class Distribution")
plt.savefig("outputs/1_class_distribution.png")

# 2 Height distribution (histogram of image height)
plt.figure()
sns.histplot(df["height"],kde=True)
plt.savefig("outputs/2_height.png")

# 3 Width distribution (histogram of image width)
plt.figure()
sns.histplot(df["width"],kde=True)
plt.savefig("outputs/3_width.png")

# 4 Mean (brighteness) pixel distribution  (shows lighting distribution)
plt.figure()
sns.histplot(df["mean"],kde=True)
plt.savefig("outputs/4_mean.png")

# 5 Std distribution (for contrast variability)
plt.figure()
sns.histplot(df["std"],kde=True)
plt.savefig("outputs/5_std.png")

# 6 Boxplot mean per class (brightness difference between classes)
plt.figure(figsize=(12,5))
sns.boxplot(x="label",y="mean",data=df)
plt.xticks(rotation=90)
plt.savefig("outputs/6_box_mean.png")

# 7 Violin (shows spread of contrast per class)
plt.figure(figsize=(12,5))
sns.violinplot(x="label",y="std",data=df)
plt.xticks(rotation=90)
plt.savefig("outputs/7_violin_std.png")

# 8 Correlation heatmap (for relationship between variables)
plt.figure()
sns.heatmap(df[["height","width","mean","std"]].corr(),annot=True)
plt.savefig("outputs/8_corr.png")

# 9 Scatter mean vs std (mean vs std colored by class)
plt.figure()
sns.scatterplot(x="mean",y="std",hue="label",data=df,legend=False)
plt.savefig("outputs/9_scatter.png")

# 10 Pairplot (ploting every feature against every other feature)
sns.pairplot(df[["height","width","mean","std"]])
plt.savefig("outputs/10_pair.png")

# PCA for clustering
X=df[["height","width","mean","std"]]   #selects numerical features
pca=PCA(n_components=2)   #reduce to 2 dimensions
X2=pca.fit_transform(X)   #transforms data into compression version

# 11 PCA plot     #vsualizes data structure in 2D
plt.figure()
plt.scatter(X2[:,0],X2[:,1])
plt.title("PCA Projection")
plt.savefig("outputs/11_pca.png")

# KMeans clustering    groups images into 10 clusters
kmeans=KMeans(n_clusters=10)
clusters=kmeans.fit_predict(X)  #assigning cluster number to each image

# 12 Cluster plot
plt.figure()
plt.scatter(X2[:,0],X2[:,1],c=clusters)
plt.savefig("outputs/12_clusters.png")

# t-SNE   maps data to 2D while preserving the structure
tsne=TSNE(n_components=2,perplexity=30)
X3=tsne.fit_transform(X)

# 13 TSNE   #shows hidden patterns visually
plt.figure()
plt.scatter(X3[:,0],X3[:,1],c=clusters)
plt.savefig("outputs/13_tsne.png")

# More distributions

# 14 CDF shows cumulative brightness distribution
plt.figure()
df["mean"].hist(cumulative=True,bins=50)
plt.savefig("outputs/14_cdf.png")

# 15 Density   for smoth distribution curve
plt.figure()
df["mean"].plot(kind="density")
plt.savefig("outputs/15_density.png")

# 16 Countplot is a class count chart
plt.figure(figsize=(12,5))
sns.countplot(x="label",data=df)
plt.xticks(rotation=90)
plt.savefig("outputs/16_count.png")

# 17 KDE is a smoothed std distribution
plt.figure()
sns.kdeplot(df["std"])
plt.savefig("outputs/17_kde.png")

# 18 Histogram stacked is a histogram for all numeric feature
plt.figure()
df.hist(figsize=(8,6))
plt.savefig("outputs/18_multi_hist.png")

# 19 Area plot  sorted brightness as areas chat
plt.figure()
df["mean"].sort_values().reset_index(drop=True).plot.area()
plt.savefig("outputs/19_area.png")

# 20 Line plot
plt.figure()
df["mean"].sort_values().plot()
plt.savefig("outputs/20_line.png")

# 21 Rolling mean   moving average of brightness
plt.figure()
df["mean"].rolling(100).mean().plot()
plt.savefig("outputs/21_roll.png")

# 22 Box all  mean vs std spread
plt.figure()
sns.boxplot(data=df[["mean","std"]])
plt.savefig("outputs/22_box_all.png")

# 23 Jointplot   is a combination of scatter and histogram
sns.jointplot(x="mean",y="std",data=df)
plt.savefig("outputs/23_joint.png")

# 24 Hexbin   density scatter
plt.figure()
plt.hexbin(df["mean"],df["std"],gridsize=30)
plt.savefig("outputs/24_hex.png")

# 25 Final distribution    high resolution brighness distribution
plt.figure()
sns.histplot(df["mean"],bins=100)
plt.savefig("outputs/25_final.png")

print("Done. All plots saved in outputs folder.")


#This helps you verify dataset quality before training a model, which prevents poor accuracy later.