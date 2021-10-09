#IMPORTES :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyclustertend import hopkins
from sklearn import metrics
from sklearn.model_selection import train_test_split

 # - - - - -  - - - - - - - - -  - - - - - - - - - -  - 
# DATA FRAME :
url = 'https://gist.githubusercontent.com/MaaniBeigy/3f55fcc77551b9c2218a9bf19e800f47/raw/8affb3ae2c9fe1725878b0cdc43dfb8da31ba164/df_total.csv'
df = pd.read_csv(url,index_col=None)

df["Temperature"] = StandardScaler().fit_transform(df[["Temperature"]])
df["GSR"] = StandardScaler().fit_transform(df[["GSR"]])
df["EOG1"] = StandardScaler().fit_transform(df[["EOG1"]])
df["EOG2"] = StandardScaler().fit_transform(df[["EOG2"]])
df["EEG1"] = StandardScaler().fit_transform(df[["EEG1"]])
df["EEG2"] = StandardScaler().fit_transform(df[["EEG2"]])
df["RED_RAW"] = StandardScaler().fit_transform(df[["RED_RAW"]])
df["IR_RAW"] = StandardScaler().fit_transform(df[["IR_RAW"]])

x = df[[
        'Temperature', 'GSR', 'EOG1', 'EOG2', 'EEG1', 'EEG2', 'RED_RAW',
       'IR_RAW']].to_numpy()

X_train, X_test = train_test_split(x, test_size=0.5, random_state=0)

# - - - - -  - - - - - - - - -  - - - - - - - - - -  - 

# METRICS : 

df.shape
H = hopkins(x,64074) #Result is : 0.0034266305188508143 -> Datas are uniformly distibuted


# - - - - -  - - - - - - - - -  - - - - - - - - - -  - 
# TODO : Fixing These codes : 

# df_cl = df.columns

# def st_vec(input):
#     for i in df_cl :
#         df['%s_zscore' %i] = StandardScaler().fit_transform(input)

# for vec in df_cl : 
#     x_f = df[["%s" %vec]]
#     st_vec(x_f)

# df.drop(columns=['Temperature', 'GSR', 'EOG1', 'EOG2', 'EEG1', 'EEG2', 'RED_RAW',
#        'IR_RAW','Arousal','Dominance','Valence'])

# - - - - -  - - - - - - - - -  - - - - - - - - - -  - 

# ELBOW MET 1 :

# distortions = []
# for i in range(1, 11):
#     km = KMeans(
#         n_clusters=i, init='k-means++',
#         n_init=10, max_iter=300,
#         tol=1e-04, random_state=0
#     )
#     km.fit(x)
#     distortions.append(km.inertia_)

# # plot
# plt.plot(range(1, 11), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()
 
 # The number of clusters is 3 USING Elbow-Met

 # - - - - -  - - - - - - - - -  - - - - - - - - - -  - 

 # ELBOW MET 2 :

from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,11))

visualizer.fit(x)        # Fit the data to the visualizer
#visualizer.show()        # Finalize and render the figure
# With this method the numbers of clusters was 4

 # - - - - -  - - - - - - - - -  - - - - - - - - - -  - 

kmeans_model = KMeans(n_clusters=4, random_state=1,init='k-means++').fit(X_train)
labels = kmeans_model.labels_

kmeans_model_pred = KMeans(n_clusters=4, random_state=1,init='k-means++').fit(X_test)
labels_pred = kmeans_model_pred.labels_
 # - - - - -  - - - - - - - - -  - - - - - - - - - -  - 

# METRICS : 

H = hopkins(x,64074) #Result is : 0.0034266305188508143 -> Datas are uniformly distibuted

# TODO : Fix lables : 
clustering_metrics = [
    metrics.calinski_harabasz_score(X_train, labels), #9377.479444286533
    metrics.homogeneity_score(labels, labels_pred), #0.00017892611880187027
    metrics.rand_score(labels, labels_pred),#0.5083643240212956
    metrics.davies_bouldin_score(X_train, labels), #1.1200761594594308
    metrics.completeness_score(labels, labels_pred), #0.0001796496426566734
    metrics.silhouette_score(X_train, labels), #0.2902319948950943
]