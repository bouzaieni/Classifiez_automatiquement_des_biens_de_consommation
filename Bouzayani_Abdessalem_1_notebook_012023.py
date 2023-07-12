#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import manifold, decomposition

from sklearn import cluster, metrics
from sklearn.decomposition import NMF, LatentDirichletAllocation


import tensorflow as tf
import tensorflow.keras
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.densenet import DenseNet201
from keras.applications.densenet import preprocess_input


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import gensim


import tensorflow_hub as hub

import os
import transformers
from transformers import *


from PIL import Image, ImageOps, ImageFilter
import cv2
import time

import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


debut_notebook = time.time()


# # Jeu de données

# In[3]:


data = pd.read_csv('flipkart_com-ecommerce_sample_1050.csv')
print(data.shape)
data.sample(5)
data_p6 = data.copy()


# In[4]:


print(data['product_category_tree'][0])


# In[5]:


data['product_name'][0]


# In[6]:


#Extraction des catégories
def extraction_categries(categ, level):
    
  categ = categ.split('["')[1].split('"]')[0]
  cat = categ.split(' >> ')

  if(len(cat)) < 3:
    cat = [cat[0], cat[1], 'Nan']
      
    if(len(cat))<2:
      cat = [cat[0], 'Nan', 'Nan']

  return cat[level]

def categ_1(categ):
  return extraction_categries(categ,0)

def categ_2(categ):
  return extraction_categries(categ,1)

def categ_3(categ):
  return extraction_categries(categ,2)

def tree_categ(dataframe):

  dataframe['categ_level_1'] = dataframe['product_category_tree'].apply(categ_1)
  dataframe['categ_level_2'] = dataframe['product_category_tree'].apply(categ_2)
  dataframe['categ_level_3'] = dataframe['product_category_tree'].apply(categ_3)

  return dataframe.drop(['product_category_tree'], axis=1)


# In[7]:


data = tree_categ(data)


# In[8]:


# nbre de categories par niveau
print('nbre de catégories niveau 1 : ',data['categ_level_1'].nunique())

print('nbre de catégories niveau 2 : ',data['categ_level_2'].nunique())


print('nbre de catégories niveau 3 : ',data['categ_level_3'].nunique())


# In[9]:


data.head()


# In[10]:


data['label']= LabelEncoder().fit_transform(np.array(data['categ_level_1']))


# In[11]:


data.head(1)


# In[12]:


data_netoyee=pd.DataFrame()
data_netoyee['image']=data['image']
data_netoyee['description']=data['description']
data_netoyee['categorie']=data['categ_level_1']
data_netoyee['label']=data['label']

data_netoyee.head()


# # Traitement d'images

# In[13]:


from os import listdir
path = 'Images/'
list_photos = [file for file in listdir(path)]
print(len(list_photos))


# In[14]:


data_netoyee.groupby("label").count()


# ## Affichage d'exemples d'images

# In[15]:


# Sélection de 9 images au hasard
df = data_netoyee.sample(9).reset_index()
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    name = path + df['image'][i]
    img = np.array(Image.open(name))
    plt.imshow(img)
    #plt.title(df['categorie'][i], fontsize=20)
    plt.axis('off')


# ## Exemple de pré-traitement sur une image

# ### Histogramme

# In[16]:


img_orig = Image.open(path + 'aadbc3b9c32c4535b1bfee7321c4c0e7.jpg')
img_test = img_orig

dim_test = img_test.size
plt.figure(figsize=(30, 10))
plt.subplot(121)
plt.title('Exemple d\'une image', fontsize=30)
plt.imshow(img_test, cmap='gray')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.subplot(122)
plt.title('Histogramme', fontsize=30)
hist, bins = np.histogram(np.array(img_test).flatten(), bins=256)
plt.bar(range(len(hist[0:255])), hist[0:255])
plt.xlabel('Niveau de gris', fontsize=30)
plt.ylabel('Nombre de pixels', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# ### Étirement d’histogrammes

# In[17]:


img_test = ImageOps.autocontrast(img_test, 1)
plt.figure(figsize=(30, 10))
plt.subplot(121)
plt.title('Exemple d\'une image', fontsize=30)
plt.imshow(img_test, cmap='gray')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.subplot(122)
plt.title('Etirement de l`\'histogramme', fontsize=30)
hist, bins = np.histogram(np.array(img_test).flatten(), bins=256)
plt.bar(range(len(hist[0:255])), hist[0:255])
plt.xlabel('Niveau de gris', fontsize=30)
plt.ylabel('Nombre de pixels', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# ### Égalisation d’histogrammes

# In[18]:


img_test = ImageOps.equalize(img_test)
plt.figure(figsize=(30, 10))
plt.subplot(121)
plt.title('Exemple d\'une image', fontsize=30)
plt.imshow(img_test, cmap='gray')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.subplot(122)
plt.title('Egalisation de l\'histogramme', fontsize=30)
hist, bins = np.histogram(np.array(img_test).flatten(), bins=256)
plt.bar(range(len(hist[0:255])), hist[0:255])
plt.xlabel('Niveau de gris', fontsize=30)
plt.ylabel('Nombre de pixels', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# ### Image en niveau de gris

# In[19]:


img_gris = cv2.cvtColor(np.array(img_test), cv2.COLOR_RGB2GRAY)
plt.figure(figsize=(30, 10))
plt.subplot(121)
plt.title('Image en niveau de gris', fontsize=30)
plt.imshow(img_gris, cmap='gray')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.subplot(122)
plt.title('Histogramme', fontsize=30)
hist, bins = np.histogram(np.array(img_gris).flatten(), bins=256)
plt.bar(range(len(hist[0:255])), hist[0:255])
plt.xlabel('Niveau de gris', fontsize=30)
plt.ylabel('Nombre de pixels', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# ### Élimination du bruit

# In[20]:


img_test = cv2.resize(np.array(img_test), (224, 224),
                            interpolation=cv2.INTER_AREA)
img_test = Image.fromarray(img_test, 'RGB')
img_bruit = img_test.filter(ImageFilter.BoxBlur(1))

plt.figure(figsize=(30, 10))
plt.subplot(121)
plt.title('Image avec bruit', fontsize=30)
plt.imshow(img_test, cmap='gray')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.subplot(122)
plt.title('Image sans bruit', fontsize=30)
plt.imshow(img_bruit, cmap='gray')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# ### Redimensionnement

# In[21]:


img_redim = cv2.resize(np.array(img_test), (100, 100),
                             interpolation=cv2.INTER_AREA)
plt.figure(figsize=(30, 10))
plt.subplot(121)
plt.title('Image originale', fontsize=30)
plt.imshow(img_test, cmap='gray')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.subplot(122)
plt.title('Image redimentionnée', fontsize=30)
plt.imshow(img_redim, cmap='gray')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# ## SIFT

# Détermination et affichage des descripteurs SIFT

# In[22]:


img_test = path + 'aadbc3b9c32c4535b1bfee7321c4c0e7.jpg'
sift = cv2.xfeatures2d.SIFT_create()
image = cv2.imread(img_test,0) # convert in gray
image = cv2.equalizeHist(image)   # equalize image histogram
kp, des = sift.detectAndCompute(image, None)
img=cv2.drawKeypoints(image,kp,image)
plt.imshow(img)
plt.show()
print("Descripteurs : ", des.shape)
print()
print(des)


# Créations des descripteurs de chaque image
# * Pour chaque image passage en gris et equalisation
# * création d'une liste de descripteurs par image ("sift_keypoints_by_img") qui sera utilisée pour réaliser les histogrammes par image
# * création d'une liste de descripteurs pour l'ensemble des images ("sift_keypoints_all") qui sera utilisé pour créer les clusters de descripteurs

# In[23]:


list_photos = data_netoyee['image'].values.tolist()


# In[24]:


'''
# identification of key points and associated descriptors
import time, cv2
sift_keypoints = []
temps1=time.time()
sift = cv2.xfeatures2d.SIFT_create()

for image_num in range(len(list_photos)) :
    if image_num%100 == 0 : print(image_num)
    image = cv2.imread(path+list_photos[image_num],0) # convert in gray
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    res = cv2.equalizeHist(image)   # equalize image histogram
    kp, des = sift.detectAndCompute(res, None)
    sift_keypoints.append(des)
    

sift_keypoints_by_img = sift_keypoints
#sift_keypoints_by_img = np.asarray(sift_keypoints)

sift_keypoints_all    = np.concatenate(sift_keypoints_by_img, axis=0)

#del sift_keypoints_by_img

print()
print("Nombre de descripteurs : ", sift_keypoints_all.shape)

duration1=time.time()-temps1
print("temps de traitement SIFT descriptor : ", "%15.2f" % duration1, "secondes")
'''


# Le calcul des descripteurs sift (code dans la cellule précédente) prend du temps. J'ai préféré calculer ces descripteurs et les enregistrer une fois pour toute.

# In[25]:


#np.save('sift_keypoints_by_img.npy', sift_keypoints_by_img) # save


# In[26]:


sift_keypoints_by_img = np.load('sift_keypoints_by_img.npy', allow_pickle=True) # save


# In[27]:


sift_keypoints_all = np.concatenate(sift_keypoints_by_img, axis=0)


#  Création des clusters de descripteurs
# * Utilisation de MiniBatchKMeans pour obtenir des temps de traitement raisonnables

# In[28]:


# Determination number of clusters
temps1=time.time()

#k = int(round(np.sqrt(len(sift_keypoints_all)),0))
# k = nbre_categorie*10=70
k = 70
print("Nombre de clusters estimés : ", k)
print("Création de",k, "clusters de descripteurs ...")

# Clustering
kmeans = cluster.MiniBatchKMeans(n_clusters=k, init_size=3*k, random_state=42)

kmeans.fit(sift_keypoints_all)

duration1=time.time()-temps1
print("temps de traitement kmeans : ", "%15.2f" % duration1, "secondes")


#  Création des features des images
# * Pour chaque image : 
#    - prédiction des numéros de cluster de chaque descripteur
#    - création d'un histogramme = comptage pour chaque numéro de cluster du nombre de descripteurs de l'image
#    
#   Features d'une image = Histogramme d'une image = Comptage pour une image du nombre de descripteurs par cluster

# In[29]:


# Creation of histograms (features)
temps1=time.time()

def build_histogram(kmeans, des, image_num):
    res = kmeans.predict(des)
    hist = np.zeros(len(kmeans.cluster_centers_))
    nb_des=len(des)
    if nb_des==0 : print("problème histogramme image  : ", image_num)
    for i in res:
        hist[i] += 1.0/nb_des
    return hist


# Creation of a matrix of histograms
hist_vectors=[]

for i, image_desc in enumerate(sift_keypoints_by_img) :
    if i%100 == 0 : print(i)  
    hist = build_histogram(kmeans, image_desc, i) #calculates the histogram
    hist_vectors.append(hist) #histogram is the feature vector

im_features = np.asarray(hist_vectors)

duration1=time.time()-temps1
print("temps de création histogrammes : ", "%15.2f" % duration1, "secondes")


#  Réductions de dimension

# Réduction de dimension PCA
# * La réduction PCA permet de créer des features décorrélées entre elles, et de diminuer leur dimension, tout en gardant un niveau de variance expliquée élevé (99%)
# * L'impact est une meilleure séparation des données via le T-SNE et une réduction du temps de traitement du T-SNE

# In[30]:


def reduction_pca(numpy_tab):
    print("Dimensions dataset avant réduction PCA : ", numpy_tab.shape)
    pca = decomposition.PCA(n_components=0.99)
    feat_pca = pca.fit_transform(numpy_tab)
    print("Dimensions dataset après réduction PCA : ", feat_pca.shape)
    return feat_pca

def reduction_tsne(data, perp):
    tsne = manifold.TSNE(n_components=2, perplexity=perp, 
                     n_iter=2000, init='random',learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(data)

    df_tsne = pd.DataFrame(X_tsne[:,0:2], columns=['tsne1', 'tsne2'])
    df_tsne["class"] = data_netoyee["categorie"]
    #print(df_tsne.shape)
    return df_tsne


# In[31]:


# Calcul Tsne, détermination des clusters
def clustering_tsne(df_tsne) :
       
    # Détermination des clusters à partir des données après Tsne 
    X_tsne = df_tsne[['tsne1', 'tsne2']].to_numpy()

    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)
    labels_pred = cls.labels_
    df_tsne["cluster"] = labels_pred
    return labels_pred

# visualisation du Tsne selon les vraies catégories et selon les clusters
def TSNE_visu(df_tsne, labels_true, labels_pred) :
    X_tsne = df_tsne[['tsne1', 'tsne2']].to_numpy()
    fig = plt.figure(figsize=(20,10))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels_true, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=liste_categ, loc="best", title="Categorie")
    plt.title('Représentation des catégories par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels_pred, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels_pred), loc="best", title="Clusters")
    plt.title('Représentation des catégories par clusters')
    
    plt.show()


# In[32]:


feat_pca = reduction_pca(im_features)


# In[33]:


labels_true = data_netoyee["label"]
num_labels = 7
liste_categ = set(data_netoyee['categorie'])


# In[34]:


df_tsne = reduction_tsne(feat_pca, 20)


# In[35]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[ ]:





# Calcul des métriques

# In[36]:


dataframe_metrique = pd.DataFrame( columns=['Methode', 'ARI', 'homogeneity', 'completeness', 'v_measure'])


# In[37]:


def calcul_metrics_clustering(labels_true, labels_pred, methode):
    ari = np.round(metrics.adjusted_rand_score(labels_true, labels_pred), 4)
    homogeneity = np.round(metrics.homogeneity_score(labels_true, labels_pred), 4)
    completeness = np.round(metrics.completeness_score(labels_true, labels_pred), 4)
    v_measure = np.round(metrics.v_measure_score(labels_true, labels_pred), 4)
    metrique = {
                'Methode': methode,
                'ARI': ari,
                'homogeneity': homogeneity,
                'completeness':completeness,
                'v_measure': v_measure
                }

    return metrique


# In[38]:


metrique_sift = calcul_metrics_clustering(labels_true, labels_pred, 'sift')


# In[39]:


dataframe_metrique = dataframe_metrique.append(metrique_sift, ignore_index=True)


# In[40]:


dataframe_metrique


# In[41]:


df_tsne.groupby("cluster").count()["class"]


# In[42]:


conf_mat = metrics.confusion_matrix(labels_true, labels_pred)


# In[43]:


def showconfusionmatrix(conf_mat):
       
       list_labels = data_netoyee['categorie'].unique().tolist()
       df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                 columns = [i for i in "0123456"])
       plt.figure(figsize = (6,4))
       plt.title('Confusion matrix')
       sns.heatmap(df_cm, annot=True, cmap="Blues")


# In[44]:


def conf_mat_transform(y_true,y_pred) :
    conf_mat = metrics.confusion_matrix(y_true,y_pred)
    
    corresp = np.argmax(conf_mat, axis=0)
    #corresp = [6, 0, 5, 1, 3, 4, 2]
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    
    return labels['y_pred_transform']

cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# ## ORB

# In[45]:


'''
# identification of key points and associated descriptors
import time, cv2
orb_keypoints = []
temps1=time.time()
orb = cv2.ORB_create()

for image_num in range(len(list_photos)) :
    if image_num%100 == 0 : print(image_num)
    image = cv2.imread(path+list_photos[image_num],0) # convert in gray
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    res = cv2.equalizeHist(image)   # equalize image histogram
    kp, des = orb.detectAndCompute(res, None)
    orb_keypoints.append(des)
    #del kp, des ,res

orb_keypoints_by_img = np.asarray(orb_keypoints)
orb_keypoints_all    = np.concatenate(orb_keypoints_by_img, axis=0)
#del orb_keypoints_by_img

print()
print("Nombre de descripteurs : ",orb_keypoints_all.shape)

duration1=time.time()-temps1
print("temps de traitement orb descriptor : ", "%15.2f" % duration1, "secondes")
'''


# In[46]:


#np.save('orb_keypoints_by_img.npy', orb_keypoints_by_img) # save


# In[47]:


orb_keypoints_by_img = np.load('orb_keypoints_by_img.npy', allow_pickle=True) # save


# In[48]:


orb_keypoints_all = np.concatenate(orb_keypoints_by_img, axis=0)


# In[49]:


from sklearn import cluster, metrics

# Determination number of clusters
temps1=time.time()

k = int(round(np.sqrt(len(np.concatenate(orb_keypoints_by_img, axis=0))),0))
# k = nbre_categorie*10=70
print("Nombre de clusters estimés : ", k)
print("Création de",k, "clusters de descripteurs ...")

# Clustering
kmeans = cluster.MiniBatchKMeans(n_clusters=k, init_size=3*k, random_state=42)
kmeans.fit(orb_keypoints_all)

duration1=time.time()-temps1
print("temps de traitement kmeans : ", "%15.2f" % duration1, "secondes")


# In[50]:


# Creation of histograms (features)
temps1=time.time()

def build_histogram(kmeans, des, image_num):
    res = kmeans.predict(des)
    hist = np.zeros(len(kmeans.cluster_centers_))
    nb_des=len(des)
    if nb_des==0 : print("problème histogramme image  : ", image_num)
    for i in res:
        hist[i] += 1.0/nb_des
    return hist


# Creation of a matrix of histograms
hist_vectors=[]

for i, image_desc in enumerate(orb_keypoints_by_img) :
    if i%100 == 0 : print(i)  
    hist = build_histogram(kmeans, image_desc, i) #calculates the histogram
    hist_vectors.append(hist) #histogram is the feature vector

im_features = np.asarray(hist_vectors)

duration1=time.time()-temps1
print("temps de création histogrammes : ", "%15.2f" % duration1, "secondes")


# In[51]:


feat_pca = reduction_pca(im_features)


# In[52]:


df_tsne = reduction_tsne(feat_pca, 20)


# In[53]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[ ]:





# In[54]:


metrique_orb = calcul_metrics_clustering(labels_true, labels_pred, 'orb')
dataframe_metrique = dataframe_metrique.append(metrique_orb, ignore_index=True)
dataframe_metrique


# In[55]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# In[56]:


del orb_keypoints_all, sift_keypoints_all, orb_keypoints_by_img, sift_keypoints_by_img


# ## Transfer Learning

# In[57]:


from tensorflow.keras import applications


# In[58]:


dir(applications)


# ### VGG16

# In[59]:


def transfer_learn(model):



	# Liste
	model_all_features = []

	# Instanciation du modèle
	model_learn = model
	# Résumé de l'architecture du modèle
	model_learn.summary()

	for image_num in list_photos:
		image_num = path+image_num
		# Charger l'image et la redimensionner à la taille
		# requise de 224×224 pixels.
		img = keras.preprocessing.image.load_img(image_num, target_size=(224, 224))
		# Convertir les pixels en un tableau NumPy afin de pouvoir travailler
		# avec dans Keras
		img = keras.preprocessing.image.img_to_array(img)
		# Redimensionnement
		img = np.expand_dims(img, axis=0)
		# Préparer de nouvelles entrées pour le réseau.
		img = preprocess_input(img)

		# obtenir une prédiction de la probabilité d'appartenance
		# de l'image à chacun des 1000 types d'objets connus.
		model_feature = model_learn.predict(img)
		# Ajouter la feature prédite en nparray à la liste
		model_all_features.append(np.array(model_feature).flatten())

	model_all_features = np.array(model_all_features)
	return model_all_features


# In[60]:


model = VGG16(weights='imagenet', include_top=False)
vgg16_all_features = transfer_learn(model)


# In[61]:


def optimisation_perplexity(data, liste_perp):
    for perp in liste_perp:
        tsne = manifold.TSNE(n_components=2, perplexity=perp, 
                             n_iter=2000, init='random', learning_rate=200, random_state=42)
        X_tsne = tsne.fit_transform(data)

        df_tsne = pd.DataFrame(X_tsne[:,0:2], columns=['tsne1', 'tsne2'])
        df_tsne["class"] = data_netoyee["categorie"]
        cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)

        cls.fit(X_tsne)
        df_tsne["cluster"] = cls.labels_
        print("perplexity : ", perp, '-------', "ARI : ", np.round(metrics.adjusted_rand_score(labels_true, cls.labels_), 4))


# In[62]:


feat_pca = reduction_pca(vgg16_all_features)


# In[63]:


liste_perp = [10, 20, 30, 40, 50]
optimisation_perplexity(feat_pca, liste_perp)


# In[64]:


df_tsne = reduction_tsne(feat_pca, 40)


# In[ ]:





# In[65]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[66]:


metrique_vgg16 = calcul_metrics_clustering(labels_true, labels_pred, 'vgg16')
dataframe_metrique = dataframe_metrique.append(metrique_vgg16, ignore_index=True)
dataframe_metrique


# In[67]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# In[68]:


def taille_cluster(labels):
    #global cluster_size
    cluster_size = {}
    for i in labels:
        if i not in cluster_size:
            cluster_size[i] = 1
        else:
            cluster_size[i] += 1

    fig = plt.figure(figsize=[8, 8])
    fig.patch.set_alpha(0.7)
    plt.title("Clusters size ")
    plt.bar(range(0, len(cluster_size)), cluster_size.values())
    for i, num in enumerate(cluster_size.keys()):
        height = cluster_size[num]+20
        x = i - 0.25
        pourcent = cluster_size[num]/labels.shape[0]*100
        plt.text(x, height, "{} % ".format(round(pourcent, 2)))
    plt.xticks(range(0, len(cluster_size)), cluster_size.keys())


# In[69]:


taille_cluster(df_tsne["cluster"])


# In[70]:


# Les catégories par clusters
df_tsne.groupby('cluster')['class'].value_counts().     to_frame()


# ### MobileNetV2

# In[71]:


model = MobileNetV2(weights='imagenet', include_top=False)
mob_all_features = transfer_learn(model)


# In[72]:


feat_pca = reduction_pca(mob_all_features)


# In[73]:


optimisation_perplexity(feat_pca, liste_perp)


# In[74]:


df_tsne = reduction_tsne(feat_pca, 30)


# In[75]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[76]:


metrique_mobnet = calcul_metrics_clustering(labels_true, labels_pred, 'mobnet')
dataframe_metrique = dataframe_metrique.append(metrique_mobnet, ignore_index=True)
dataframe_metrique


# In[77]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# In[78]:


taille_cluster(df_tsne["cluster"])


# In[79]:


# Les catégories par clusters
df_tsne.groupby('cluster')['class'].value_counts().     to_frame()


# ### InceptionV3

# In[80]:


model = InceptionV3(weights='imagenet', include_top=False)
incep_all_features = transfer_learn(model)


# In[81]:


feat_pca = reduction_pca(incep_all_features)


# In[82]:


optimisation_perplexity(feat_pca, liste_perp)


# In[83]:


df_tsne = reduction_tsne(feat_pca, 50)


# In[84]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[85]:


metrique_inception = calcul_metrics_clustering(labels_true, labels_pred, 'inception')
dataframe_metrique = dataframe_metrique.append(metrique_inception, ignore_index=True)
dataframe_metrique


# In[86]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# In[87]:


taille_cluster(df_tsne["cluster"])


# In[88]:


# Les catégories par clusters
df_tsne.groupby('cluster')['class'].value_counts().     to_frame()


# ### DenseNet

# In[89]:


model = DenseNet201(weights='imagenet', include_top=False)
dense_all_features = transfer_learn(model)


# In[90]:


feat_pca = reduction_pca(dense_all_features)


# In[91]:


optimisation_perplexity(feat_pca, liste_perp)


# In[92]:


df_tsne = reduction_tsne(feat_pca, 20)


# In[93]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[94]:


metrique_dense = calcul_metrics_clustering(labels_true, labels_pred, 'dense')
dataframe_metrique = dataframe_metrique.append(metrique_dense, ignore_index=True)
dataframe_metrique


# In[95]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# In[96]:


taille_cluster(df_tsne["cluster"])


# In[97]:


# Les catégories par clusters
df_tsne.groupby('cluster')['class'].value_counts().     to_frame()


# ### InceptionResNetV2

# In[98]:


from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

model = InceptionResNetV2(weights='imagenet', include_top=False)
incepres_all_features = transfer_learn(model)


# In[99]:


feat_pca = reduction_pca(incepres_all_features)


# In[100]:


optimisation_perplexity(feat_pca, liste_perp)


# In[101]:


df_tsne = reduction_tsne(feat_pca, 20)


# In[102]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[103]:


metrique_incepres= calcul_metrics_clustering(labels_true, labels_pred, 'incepresn')
dataframe_metrique = dataframe_metrique.append(metrique_incepres, ignore_index=True)
dataframe_metrique


# In[104]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# In[105]:


taille_cluster(df_tsne["cluster"])


# In[106]:


# Les catégories par clusters
df_tsne.groupby('cluster')['class'].value_counts().     to_frame()


# # Traitement de texte

# In[107]:


# Tokenizer

def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

# Stop words
from nltk.corpus import stopwords
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']

def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Lemmatizer (base d'un mot)
from nltk.stem import WordNetLemmatizer

def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

def transform_bow_lem_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

def transform_dl_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    lw = lower_start_fct(word_tokens)
    transf_desc_text = ' '.join(lw)
    return transf_desc_text


# In[108]:


data_nlp  = data_netoyee.copy()


# In[109]:


data_nlp.head()


# ## Exemple de pré-traitement 

# In[110]:


text_test = data_nlp['description'].head(1)
print(data_nlp['description'][0])
print('\n')
print(len(data_nlp['description'][0]))


# In[111]:


import re
#Suppresion des chiffres
text_test = (re.sub(r'\d+', '', text_test.item()))
print(text_test)
print('\n')
print(len(text_test))


# In[112]:


#suppresion des petits mots
shortword = re.compile(r'\W*\b\w{1,2}\b')
text_test=shortword.sub('', text_test)
print(text_test)
print('\n')
print(len(text_test))


# In[113]:


# Extraction des tokens
tokenizer = nltk.RegexpTokenizer(r'\w+')
text_test=tokenizer.tokenize(text_test)
print(text_test)
print('\n')
print(len(text_test))


# In[114]:


#Normalisation
text_test_low = []
for i in text_test:
    text_test_low.append(i.lower())
print(text_test_low)
print('\n')
print(len(text_test_low))


# In[115]:


#Stop word et adverbes/adjectifs

text_test_stword = stop_word_filter_fct(text_test_low)
print(text_test_stword)
print('\n')
print(len(text_test_stword))


# In[116]:


#lemmantisation

text_test_lem = lemma_fct(text_test_stword)    
print(text_test_lem)
print('\n')
print(len(text_test_lem))


# ## Pré-traitement de tout le vocabulaire

# In[117]:


data_nlp['sentence_bow'] = data_nlp['description'].apply(lambda x : transform_bow_fct(x))
data_nlp['sentence_bow_lem'] = data_nlp['description'].apply(lambda x : transform_bow_lem_fct(x))
data_nlp['sentence_dl'] = data_nlp['description'].apply(lambda x : transform_dl_fct(x))
data_nlp.shape


# In[118]:


data_nlp.head()


# In[119]:


data_nlp['length_bow'] = data_nlp['sentence_bow'].apply(lambda x : len(word_tokenize(x)))
print("max length bow : ", data_nlp['length_bow'].max())
data_nlp['length_dl'] = data_nlp['sentence_dl'].apply(lambda x : len(word_tokenize(x)))
print("max length dl : ", data_nlp['length_dl'].max())


# ## Bags of words - CountVectorizer
# 

# In[120]:


# création du bag of words (CountVectorizer)

cvect = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)

feat = 'sentence_bow_lem'
cv_fit = cvect.fit(data_nlp[feat])

cv_transform = cvect.transform(data_nlp[feat])  


# In[121]:


optimisation_perplexity(cv_transform, liste_perp)


# In[122]:


df_tsne = reduction_tsne(cv_transform, 20)


# In[123]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[124]:


metrique_countvect = calcul_metrics_clustering(labels_true, labels_pred, 'countvect')
dataframe_metrique = dataframe_metrique.append(metrique_countvect, ignore_index=True)
dataframe_metrique


# In[125]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# ##  Bags of words - Tf-idf
# 

# In[126]:


# création du bag of words (Tf-idf)

ctf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)

feat = 'sentence_bow_lem'
ctf_fit = ctf.fit(data_nlp[feat])

ctf_transform = ctf.transform(data_nlp[feat]) 


# In[127]:


optimisation_perplexity(ctf_transform, liste_perp)


# In[128]:


df_tsne = reduction_tsne(ctf_transform, 10)


# In[129]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[130]:


metrique_tfidf = calcul_metrics_clustering(labels_true, labels_pred, 'tfidf')
dataframe_metrique = dataframe_metrique.append(metrique_tfidf, ignore_index=True)
dataframe_metrique


# In[131]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# ## Word2vec

# ### Entrainement d'un nouveau modèle

# In[132]:


w2v_size=512
w2v_window=5
w2v_min_count=1
w2v_epochs=100
maxlen = 24 # adapt to length of sentences
sentences = data_nlp['sentence_bow_lem'].to_list()
sentences = [gensim.utils.simple_preprocess(text) for text in sentences]


# In[133]:


# Création et entraînement du modèle Word2Vec

print("Build & train Word2Vec model ...")
w2v_model = gensim.models.Word2Vec(min_count=w2v_min_count, window=w2v_window,
                                                vector_size=w2v_size,
                                                seed=42,
                                                workers=1)
#                                                workers=multiprocessing.cpu_count())
w2v_model.build_vocab(sentences)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_epochs)
model_vectors = w2v_model.wv
w2v_words = model_vectors.index_to_key
print("Vocabulary size: %i" % len(w2v_words))
print("Word2Vec trained")


# In[134]:


# Préparation des sentences (tokenization)

print("Fit Tokenizer ...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
x_sentences = pad_sequences(tokenizer.texts_to_sequences(sentences),
                                                     maxlen=maxlen,
                                                     padding='post') 
                                                   
num_words = len(tokenizer.word_index) + 1
print("Number of unique words: %i" % num_words)


# In[135]:


# Création de la matrice d'embedding

print("Create Embedding matrix ...")
w2v_size = 512
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, w2v_size))
i=0
j=0
    
for word, idx in word_index.items():
    i +=1
    if word in w2v_words:
        j +=1
        embedding_vector = model_vectors[word]
        if embedding_vector is not None:
            embedding_matrix[idx] = model_vectors[word]
            
word_rate = np.round(j/i,4)
print("Word embedding rate : ", word_rate)
print("Embedding matrix: %s" % str(embedding_matrix.shape))


# In[136]:


# Création du modèle

input=Input(shape=(len(x_sentences),maxlen),dtype='float64')
word_input=Input(shape=(maxlen,),dtype='float64')  
word_embedding=Embedding(input_dim=vocab_size,
                         output_dim=w2v_size,
                         weights = [embedding_matrix],
                         input_length=maxlen)(word_input)
word_vec=GlobalAveragePooling1D()(word_embedding)  
embed_model = Model([word_input],word_vec)

embed_model.summary()


# In[137]:


embeddings = embed_model.predict(x_sentences)
embeddings.shape


# In[138]:


optimisation_perplexity(embeddings, liste_perp)


# In[139]:


df_tsne = reduction_tsne(embeddings, 40)


# In[140]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[141]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# ### Utilisation d'un modèle pré-entrainé

# In[142]:


import gensim.downloader as gen_dow
word_2_vec = gen_dow.load("word2vec-google-news-300")  


# In[143]:


word_2_vec.vectors.shape


# In[144]:


def doc_to_vec(document,model):
    size = model.vectors.shape[1]
    vecteur = np.zeros(size)
    nbre_token = 0
  
    for token in document:
        
        try:
            values = model[token]
            vecteur = vecteur + values
            nbre_token = nbre_token + 1.0
        except:
            pass
    
    if (nbre_token > 0.0):
        vecteur = vecteur/nbre_token
    return vecteur


# In[145]:


sentences = [sent.split() for sent in data_nlp['sentence_dl'].to_list()]


# In[146]:


docsVec = list()
for document in sentences:
    #calcul du vecteur
    vec = doc_to_vec(document, word_2_vec)
    docsVec.append(vec)
embeddings = np.array(docsVec)
print(embeddings.shape)


# In[147]:


feat_pca = reduction_pca(embeddings)


# In[148]:


optimisation_perplexity(feat_pca, liste_perp)


# In[149]:


df_tsne = reduction_tsne(feat_pca, 20)


# In[150]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[151]:


metrique_word2vec = calcul_metrics_clustering(labels_true, labels_pred, 'word2vec')
dataframe_metrique = dataframe_metrique.append(metrique_word2vec, ignore_index=True)
dataframe_metrique


# In[152]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# ## BERT

# In[153]:


# Fonction de préparation des sentences
def bert_inp_fct(sentences, bert_tokenizer, max_length) :
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens = True,
                                              max_length = max_length,
                                              padding='max_length',
                                              return_attention_mask = True, 
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0], 
                             bert_inp['token_type_ids'][0], 
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
    
    return input_ids, token_type_ids, attention_mask, bert_inp_tot
    

# Fonction de création des features
def feature_BERT_fct(model, model_type, sentences, max_length, b_size, mode='HF') :
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx+batch_size], 
                                                                      bert_tokenizer, max_length)
        
        if mode=='HF' :    # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode=='TFhub' : # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids" : input_ids, 
                                 "input_mask" : attention_mask, 
                                 "input_type_ids" : token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']
             
        if step ==0 :
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else :
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot,last_hidden_states))
    
    features_bert = np.array(last_hidden_states_tot).mean(axis=1)
    
    time2 = np.round(time.time() - time1,0)
    print("temps traitement : ", time2)
     
    return features_bert, last_hidden_states_tot


#  BERT HuggingFace

# In[154]:


max_length = 150
batch_size = 20
model_type = 'bert-base-uncased'
model = TFAutoModel.from_pretrained(model_type)
sentences = data_nlp['sentence_dl'].to_list()


# In[155]:


# Création des features

features_bert, last_hidden_states_tot = feature_BERT_fct(model, model_type, sentences, 
                                                         batch_size, max_length, mode='HF')


# In[156]:


feat_pca = reduction_pca(features_bert)


# In[157]:


optimisation_perplexity(feat_pca, liste_perp)


# In[158]:


df_tsne = reduction_tsne(feat_pca, 40)


# In[159]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[160]:


metrique_bert = calcul_metrics_clustering(labels_true, labels_pred, 'bert')
dataframe_metrique = dataframe_metrique.append(metrique_bert, ignore_index=True)
dataframe_metrique


# In[161]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# In[ ]:





# ## USE - Universal Sentence Encoder

# In[162]:


embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')


# In[163]:


def feature_USE_fct(sentences, b_size) :
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    time2 = np.round(time.time() - time1,0)
    return features


# In[164]:


batch_size = 10
sentences = data_nlp['sentence_dl'].to_list()


# In[165]:


features_USE = feature_USE_fct(sentences, batch_size)


# In[166]:


feat_pca = reduction_pca(features_USE)


# In[167]:


optimisation_perplexity(feat_pca, liste_perp)


# In[168]:


df_tsne = reduction_tsne(feat_pca, 10)


# In[169]:


labels_pred = clustering_tsne(df_tsne)
TSNE_visu(df_tsne, labels_true, labels_pred)


# In[170]:


metrique_use = calcul_metrics_clustering(labels_true, labels_pred, 'use')
dataframe_metrique = dataframe_metrique.append(metrique_use, ignore_index=True)
dataframe_metrique


# In[171]:


cls_labels_transform = conf_mat_transform(labels_true, labels_pred)
conf_mat = metrics.confusion_matrix(labels_true, cls_labels_transform)
showconfusionmatrix(conf_mat)


# # Dataframe récapitulatif

# In[172]:


plt.figure(figsize = (18, 6))

plt.bar(range(len(dataframe_metrique)), list(dataframe_metrique['ARI'].values))
plt.xticks(range(len(dataframe_metrique)), list(dataframe_metrique['Methode'].values))
plt.ylabel('Adjusted Rand Index (ARI)')
plt.title('Tested methods')
plt.show()


# In[ ]:





# In[173]:


fin_notebook = time.time()
print((fin_notebook - debut_notebook)/60)

