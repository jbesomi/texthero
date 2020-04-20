from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualization:

    @staticmethod
    def pca(data, n_components=2, scaled=True):
        if scaled:
            data = StandardScaler().fit_transform(data)  # normalizing the features
        pca = PCA(n_components).fit_transform(data)
        return pca

    @staticmethod
    def visualize(pca, classes="black"):
        plt.scatter(pca[:, 0], pca[:, 1], c=classes, s=7)
        # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap="viridis")
        plt.show()

    @staticmethod
    def visualize3D(pca, classes="black"):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax = Axes3D(fig)
        ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=classes, s=7)
        plt.show()

    @staticmethod
    def top_features_cluster(tf_idf, prediction, n_features):
        labels = np.unique(prediction)
        dfs = []
        for label in labels:
            id_temp = np.where(prediction == label)  # indices for each cluster
            x_means = np.mean(
                tf_idf[id_temp], axis=0
            )  # returns average score across cluster
            sorted_means = np.argsort(x_means)[::-1][
                :n_features
            ]  # indices with top 20 scores
            features = tf_idf.get_feature_names()
            best_features = [(features[i], x_means[i]) for i in sorted_means]
            df = pd.DataFrame(best_features, columns=["features", "score"])
            dfs.append(df)
        return dfs
