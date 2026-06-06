import numpy as np
import pandas as pd
import kagglehub
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import  accuracy_score as acs, confusion_matrix as cm, f1_score as f1
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

# Download latest version
path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")

#print("Path to dataset files:", path)
data= pd.read_csv(os.path.join(path, "Crop_recommendation.csv"))
#explorer les données
#print(data.shape)
#print(data.dtypes)
#print(data.isnull().sum())
#print(data.describe())
#print(data['label'].value_counts())
#min 0, max 140 est ce bon?
#création de la colonne N/P
#print(data["humidity"])
data["ratio_N_P"]= data["N"]/data["P"].replace(0,float("NAN"))
data["ratio_N_K"]= data["N"]/data["K"].replace(0,float("NAN"))
data["classe_temp"]= pd.cut(data["temperature"], bins=np.linspace(5,45,4), labels=["Froid", "Modéré", "Chaud"])
data["classe_humi"]= pd.cut(data["humidity"], bins=np.linspace(13,100,4), labels=["sec", "humide", "Très humide"])
#print(data.head())
#print(data["classe_temp"].value_counts(dropna=False))
#print(data["classe_humi"].value_counts(dropna=False))

# plt.show()
# plt.figure(figsize=(14,6))
# var= ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
# for i in var:
#  sns.boxplot(data=data, x= i, y="label")
#  plt.show()

# plt.figure(figsize=(14,6))
# sns.heatmap(data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].corr(), annot=True)
# plt.show()

x= data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y= data["label"]
sc= StandardScaler()
x_train, x_test, y_train, y_test= train_test_split (x,y, test_size=0.2, random_state=42)
x_train_norm= sc.fit_transform(x_train)
x_test_norm= sc.transform(x_test)
kc= KNeighborsClassifier(n_neighbors=5)
model= kc.fit(x_train_norm, y_train)
y_pred= model.predict(x_test_norm)
print(acs(y_test, y_pred))
print(f1(y_test, y_pred, average='weighted'))
print(cm(y_test, y_pred))

tree= DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=42)
tree.fit(x_train, y_train)
y_predtree= tree.predict(x_test)
print(f"accuray= {acs(y_test, y_predtree)}")
print("f1_score=" ,f1(y_test, y_predtree, average='weighted'))
print("confusion matrix" ,cm(y_test, y_predtree))

plt.figure(figsize=(16,4))
importance= tree.feature_importances_
plt.barh(["N", "P", "K", "temperature", "humidity", "ph", "rainfall"], importance)
plt.title("Attributs Important")
plt.show()

forest= RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)
y_predforest= forest.predict(x_test)
print("accuracy:", acs(y_test, y_predforest))
print("f1_score:", f1(y_test,y_predforest, average="weighted"))
print("confusion matrix:", cm(y_test,y_predforest))
plt.figure(figsize= (16,4))
importanceF= forest.feature_importances_
plt.barh(["N", "P", "K", "temperature", "humidity", "ph", "rainfall"], importanceF)
plt.show()
      












