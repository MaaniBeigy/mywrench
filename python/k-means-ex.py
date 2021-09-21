import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

url = 'https://gist.githubusercontent.com/MaaniBeigy/3f55fcc77551b9c2218a9bf19e800f47/raw/8affb3ae2c9fe1725878b0cdc43dfb8da31ba164/df_total.csv'
df = pd.read_csv(url,index_col=None)

x = df[[
        'Temperature', 'GSR', 'EOG1', 'EOG2', 'EEG1', 'EEG2', 'RED_RAW',
       'IR_RAW'
        ]].to_numpy()

y = (df['Arousal'] + df['Dominance'] + df['Valence']).to_numpy

"""
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='k-means++',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(x)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
"""
 # The number of clusters is 6 

km_main = KMeans(
    n_clusters=6, init='k-means++',
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

plt.scatter(
    x[y_km == 3, 0], x[y_km == 3, 1],
    s=50, c='purpules',
    marker='s', edgecolor='black',
    label='cluster 4'
)

plt.scatter(
    x[y_km == 4, 0], x[y_km == 4, 1],
    s=50, c='red',
    marker='s', edgecolor='black',
    label='cluster 5'
)

plt.scatter(
    x[y_km == 5, 0], x[y_km == 5, 1],
    s=50, c='yellow',
    marker='s', edgecolor='black',
    label='cluster 6'
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

