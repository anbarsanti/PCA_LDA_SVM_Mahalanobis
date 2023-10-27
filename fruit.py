# Bismillah, 23 October 2023
# EE6222 Assignment

import numpy as np 
import cv2
import glob
import os
import sys
import matplotlib.pyplot as plt
import string
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.svm import SVC
from array import array
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Get the list of all files and directories 
# in the root directory 
path = "/fruits-360"
dir_list = os.listdir(path) 
  
print("Files and directories in '", path, "' :")  
  
# print the list 
print(dir_list) 
print(os.listdir(path))

dim = 100

def getYourFruits(fruits, data_type, print_n=False, k_fold=False):
    images = []
    labels = []
    val = ['Training', 'Test']
    if not k_fold:
        path = "/fruits-360/" + data_type + "/"
        for i,f in enumerate(fruits):
            p = path + f
            j=0
            for image_path in glob.glob(os.path.join(p, "*.jpg")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (dim, dim))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                images.append(image)
                labels.append(i)
                j+=1
            if(print_n):
                print("There are " , j , " " , data_type.upper(), " images of " , fruits[i].upper())
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    else:
        for v in val:
            path = "/fruits-360/" + v + "/"
            for i,f in enumerate(fruits):
                p = path + f
                j=0
                for image_path in glob.glob(os.path.join(p, "*.jpg")):
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (dim, dim))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    images.append(image)
                    labels.append(i)
                    j+=1
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
    
def getAllFruits():
    fruits = []
    for fruit_path in glob.glob("/fruits-360/Training/*"):
        fruit = fruit_path.split("/")[-1]
        fruits.append(fruit)
    return fruits

# Choose your Fruits
fruits = ['Avocado ripe' , 'Eggplant'] #Binary classification

# Get Images and Labels 
X_train, Y_train =  getYourFruits(fruits, 'Training', print_n=True, k_fold=False)
X_test, Y_test = getYourFruits(fruits, 'Test', print_n=True, k_fold=False)

# Get data for k-fold
X,y = getYourFruits(fruits, '', print_n=True, k_fold=True)

# Scale Data Images
scaler = StandardScaler()
X_train = scaler.fit_transform([i.flatten() for i in X_train])
X_test = scaler.fit_transform([i.flatten() for i in X_test])
X = scaler.fit_transform([i.flatten() for i in X])

# Visualisation of Data
def plot_image_grid(images, nb_rows, nb_cols, figsize=(15, 15)):
    assert len(images) == nb_rows*nb_cols, "Number of images should be the same as (nb_rows*nb_cols)"
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)
    
    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].axis('off')
            axs[i, j].imshow(images[n])
            n += 1       

############# Before PCA

def getClassNumber(y):
    v =[]
    i=0
    count = 0
    for index in y:
        if(index == i):
            count +=1
        else:
            v.append(count)
            count = 1
            i +=1
    v.append(count)        
    return v

def plotPrincipalComponents(X, dim):
    v = getClassNumber(Y_train)
    colors = 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple'
    markers = ['o', 'x' , 'v', 'd']
    tot = len(X)
    start = 0 
    if(dim == 2):
        for i,index in enumerate(v):
            end = start + index
            plt.scatter(X[start:end,0],X[start:end,1] , color=colors[i%len(colors)], marker=markers[i%len(markers)], label = fruits[i])
            start = end
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    
    if(dim == 3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i,index in enumerate(v):
            end = start + index
            ax.scatter(X[start:end,0], X[start:end,1], X[start:end,2], color=colors[i%len(colors)], marker=markers[i%len(markers)], label = fruits[i])
            start = end
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')


    plt.legend(loc='lower left')
    plt.xticks()
    plt.yticks()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=fruits, yticklabels=fruits,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm,ax


############# PCA for Dimensionality Reduction --------------------------
pca = PCA(n_components=2)
X_PCA_train_2D = pca.fit_transform(X_train)
X_PCA_test_2D = pca.fit_transform(X_test)
# print("X_PCA_train_2D.shape is", X_PCA_train_2D.shape)
# print("X_PCA_test_2D.shape is", X_PCA_test_2D.shape)

# plotPrincipalComponents(X_PCA_train_2D, 2)

pca = PCA(n_components=3)
X_PCA_train_3D = pca.fit_transform(X_train)
X_PCA_test_3D = pca.fit_transform(X_test)
# print("X_PCA_train_3D.shape is", X_PCA_train_3D.shape)
# print("X_PCA_test_3D.shape is", X_PCA_test_3D.shape)
# plotPrincipalComponents(X_PCA_train_3D, 3)


############# LDA for Dimensionality Reduction --------------------------
lda = LinearDiscriminantAnalysis(n_components=1)
X_LDA = lda.fit_transform (X_train, Y_train)
X_LDA_train = lda.transform(X_train) # shape = (959, 1)
X_LDA_test = lda.transform(X_test) # shape = (322, 1)
# X_train shape is (959, 30000)
# X_test shape is (322, 30000)

############# Visualize the LDA Results --------------------------

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(X_LDA_train, Y_train, c=Y_train, cmap='rainbow', alpha=0.7, edgecolors='b')
# plt.xlabel('LDA from the Training Data Set')
# plt.ylabel('Label of the Data Set')
# plt.title('LDA for Dimensionality Reduction of Image Dataset')
# plt.colorbar(scatter, ticks=np.unique(Y_train))
# plt.show()

# plt.plot(X_LDA_train)
# plt.show()

# # ############# KERNEL SVM WITHOUT DR-----------------------------
# svm_with_kernel = SVC(gamma=0.01, kernel='rbf', probability=True)
# svm_with_kernel.fit(X_train, Y_train) 
# y_pred = svm_with_kernel.predict(X_test)
# precision = metrics.accuracy_score(y_pred, Y_test) * 100
# print("Accuracy with Kernel SVM without any DR: {0:.2f}%".format(precision))


# ##################### KERNEL SVM AFTER PCA
# svm_with_kernel = SVC(gamma=0.01, kernel='rbf', probability=True)
# svm_with_kernel.fit(X_PCA_train_2D, Y_train) 
# y_pred = svm_with_kernel.predict(X_PCA_test_2D)
# precision = metrics.accuracy_score(y_pred, Y_test) * 100
# print("Accuracy with Kernel SVM with PCA: {0:.2f}%".format(precision))

# #Plotting decision boundaries
# X_PCA_train_2D = np.int64(X_PCA_train_2D)
# X_PCA_test_2D = np.int64(X_PCA_test_2D)
# plot_decision_regions(X_PCA_train_2D, Y_train, clf=svm_with_kernel, legend=1)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('Kernel SVM Decision Boundaries with PCA')
# plt.show()

# ##################### KERNEL SVM AFTER LDA
# vm_with_kernel = SVC(gamma=0.01, kernel='rbf', probability=True)
# svm_with_kernel.fit(X_LDA_train, Y_train) 
# y_pred = svm_with_kernel.predict(X_LDA_test)
# precision = metrics.accuracy_score(y_pred, Y_test) * 100
# print("Accuracy with Kernel SVM with LDA: {0:.2f}%".format(precision))

# #Plotting decision boundaries
# X_LDA_train = np.int64(X_LDA_train)
# X_LDA_test = np.int64(X_LDA_test)
# plot_decision_regions(X_LDA_train, Y_train, clf=svm_with_kernel, legend=1)
# plt.title('Kernel SVM Decision Boundaries with LDA')
# plt.show()




#################### MAHALANOBIS

###------- DEFINE MAHALANOBIS DISTANCE FUNCTION

# class MahalanobisDistanceClassifier:
#     def fit(self, X, y):
#         self.classes = np.unique(y)
#         self.means = {label: X[y == label].mean(axis=0) for label in self.classes}
#         self.inv_cov = np.linalg.pinv(np.cov(X, rowvar=False))
#         if X.ndim == 0:
#             self.inv_cov = self.inv_cov.reshape(1, -1)

#     def predict(self, X):
#         predictions = []
#         for x in X:
#             distances = [mahalanobis(x, self.means[label], self.inv_cov) for label in self.classes]
#             predicted_class = self.classes[np.argmin(distances)]
#             predictions.append(predicted_class)
#         return np.array(predictions)
print("\n")
print("\n") 
print("\n") 

class MahalanobisDistanceClassifier:
    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.means = np.array([X[y == cls].mean(axis=0) for cls in self.classes])
        self.inv_cov = np.linalg.pinv(np.cov(X.T))
        
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dist = np.array([[mahalanobis(x, mean, self.inv_cov) for mean in self.means] for x in X])
        return self.classes[np.argmin(dist, axis=1)]


# ##################### MAHALANOBIS AFTER PCA
pca = PCA(n_components = 2)
X_MDC_train = pca.fit_transform(X_train) #training data
X_MDC_test = pca.fit_transform(X_test)   #testing data
#Y_MDC_train = pca.fit_transform(Y_train) #training label
#Y_MDC_test = pca.fit_transform(Y_test)   #testing label
print("X_MDC_train.shape is", X_LDA_train.shape)
print("Y_train.shape is", Y_train.shape)

classifier_pca = MahalanobisDistanceClassifier()
classifier_pca.fit(X_MDC_train, Y_train)
y_pred = classifier_pca.predict(X_MDC_test)
accuracy_pca = accuracy_score(Y_test, y_pred)
print(f'Accuracy with PCA: {accuracy_pca * 100:.2f}%')

plt.figure(figsize=(10, 8))
for label in np.unique(Y_test):
    mask = Y_test == label
    plt.scatter(X_test[mask, 0], X_test[mask, 1], label=f'Class {label}', alpha=0.6)

print(y_pred)
print(y_pred.shape)

X_MDC_train = np.int64(X_MDC_train)
X_MDC_test = np.int64(X_MDC_test)
plot_decision_regions(X_MDC_test, y_pred, clf=classifier_pca, legend=1)
plt.title('PCA on Mahalanobis Distance Classifier')
plt.show()


# # ##################### MAHALANOBIS AFTER LDA
# lda = LinearDiscriminantAnalysis(n_components=1)
# X_train_lda = lda.fit_transform(X_train, Y_train)
# X_test_lda = lda.fit_transform(X_test, Y_test)

# print("X_train_lda.shape is", X_LDA_train.shape)

# classifier_lda = MahalanobisDistanceClassifier()
# classifier_lda.fit(X_train_lda, Y_train)        #####!!!!!!!!!!!!!!!
# y_pred = classifier_lda.predict(X_test_lda)
# accuracy_lda = accuracy_score(Y_test, y_pred)
# print(f'Accuracy with LDA: {accuracy_lda * 100:.2f}%')