"""HW3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RiLdYwnG6bRhmbrLiNKtoTcTXznSAadB
"""

'''===Import Package==='''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.linalg import svd

from train import *

'''===Data Directory==='''
data_path = '/Users/chenguiye/Documents/大三上/機器學習/Hw3 陳祈曄 b09508004/AcromegalyFeatureSet.xlsx'
df = pd.read_excel(data_path,usecols="C:V")
newdf = df[0:103]
#print(newdf.shape)
#print(newdf.head(5))

'''===Data Normalization'''
scaled_data = preprocessing.scale(newdf)
#print(scaled_data.shape)

'''===ground truth==='''
ground_truth = pd.read_excel(data_path,usecols="B")
label = ground_truth[0:103]
scaled_label = preprocessing.scale(label)

# label encoding (remove continuous)
label_encoder = preprocessing.LabelEncoder()
encoded_label = label_encoder.fit_transform(scaled_label)

'''===Calculate Cumulative Explained Variance==='''
cov_mat = np.dot(scaled_data.T,scaled_data)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
np.round(eigen_vals,3)
np.round(eigen_vecs,3)
sum = 0
summation = eigen_vals.sum(axis = 0)
for i in range(len(eigen_vals)):
  sum += eigen_vals[i]
  if (sum/summation) >= 0.9:
    print((i+1))
    print((sum/summation))

'''===PCA==='''
pca = PCA(n_components=20)
L = pca.fit_transform(scaled_data)  # (n_samples, n_components)
print(L.shape)
cum_explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)

sns.set(style="whitegrid")  # Set Seaborn style
plt.figure(figsize=(8, 6))  # Adjust figure size if needed
sns.lineplot(x=np.arange(1, len(cum_explained_var_ratio) + 1), y=cum_explained_var_ratio)
plt.title('Cumulative Explained Variance')
plt.xlabel('# Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig("image/Cumulative Explained Variance.png")


"""Logistic model"""
from sklearn import linear_model
Logistic_Regression_model  = linear_model.LogisticRegression()
train(Logistic_Regression_model,L,encoded_label)

"""SVM model"""
from sklearn.svm import SVC
SVM_model = SVC()
train(SVM_model, L, encoded_label)

"""RandomForest model"""
from sklearn.ensemble import RandomForestClassifier
RandomForest_model = RandomForestClassifier()
train(RandomForest_model, L, encoded_label)

"""DecisionTree model"""
from sklearn.tree import DecisionTreeClassifier
DecisionTree_model = DecisionTreeClassifier()
train(DecisionTree_model, L, encoded_label)














"""ANN model"""
'''
# Commented out IPython magic to ensure Python compatibility.
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

ann_acc_array = np.empty((0,4))
ann_sensi_array = np.empty((0,4))
ann_speci_array = np.empty((0,4))

X_train, X_test, y_train, y_test = train_test_split(L, encoded_label, test_size=0.2, random_state=None)

model = Sequential()
# add hidden layer
model.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=64, kernel_initializer='normal', activation='relu'))  
# Add output layer
model.add(Dense(units=16, kernel_initializer='normal', activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=100)

# %cd '/content/gdrive/MyDrive/Colab Notebooks/機器學習/Hw3 陳祈曄 b09508004'
model.save('Ann_model')

ann = load_model('Ann_model')'''