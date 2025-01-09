import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_array = np.loadtxt('dataset.txt', dtype=int)
data_array_pca = data_array.astype(float)
data_last_100 = data_array[:, -100:]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_last_100)
pca = PCA(n_components=10)
principal_components = pca.fit_transform(data_scaled)
data_array_pca = np.hstack((data_array_pca[:, :-100], principal_components))
