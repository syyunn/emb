from sklearn.cluster import KMeans

import numpy as np

from gensim.models import KeyedVectors

embedding = KeyedVectors.load_word2vec_format("./data/gensim/embeddings.emb")


from sklearn.decomposition import PCA
import pandas as pd

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(embedding.vectors)
principalDf = pd.DataFrame(
    data=principalComponents, columns=["dim1", "dim2"]
)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

n_components = 3
plt.scatter(principalDf['dim1'], principalDf['dim2'])
plt.title(f"Example of a mixture of {n_components} distributions")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
n_clusters = 2
random_state = 0

kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
    embedding.vectors
)  # embedding.values take out ndarray

if __name__ == "__main__":
    pass
