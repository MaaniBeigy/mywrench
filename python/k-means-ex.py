#IMPORTES :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

 # - - - - -  - - - - - - - - -  - - - - - - - - - -  - 

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

# x = df[[
#         'Temperature', 'GSR', 'EOG1', 'EOG2', 'EEG1', 'EEG2', 'RED_RAW',
#        'IR_RAW']].to_numpy()

# # x_normalized = normalize(x)

# y = (df['Arousal'] + df['Dominance'] + df['Valence']).to_numpy

# - - - - -  - - - - - - - - -  - - - - - - - - - -  - 

# df_cl = df.columns

# def st_vec(input):
#     for i in df_cl :
#         df['%s_zscore' %i] = StandardScaler().fit_transform(input)

# for vec in df_cl : 
#     x_f = df[["%s" %vec]]
#     print (x_f)
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

# from sklearn.datasets import make_blobs
# from yellowbrick.cluster import KElbowVisualizer

# # Instantiate the clustering model and visualizer
# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1,11))

# visualizer.fit(x)        # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figure
# # With this method the numbers of clusters was 3

 # - - - - -  - - - - - - - - -  - - - - - - - - - -  - 

km_main = KMeans(
    n_clusters=3, init='k-means++',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km_main.fit_predict(x)

plt.scatter(
    x[y_km == 0, 0], x[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    x[y_km == 1, 0], x[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)


plt.scatter(
    x[y_km == 2, 0], x[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)
# plot the centroids

plt.scatter(
    km_main.cluster_centers_[:, 0], km_main.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

