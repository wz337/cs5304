#Iris Zhang (wz337)
#Dexing Xu (dx47)

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_rcv1


def load_labels(path_to_labels):
    labels = pd.read_csv(path_to_labels, names=['label'], dtype=np.int32)
    return labels['label'].tolist()


def load_training_data():
    data = fetch_rcv1(subset='train')
    return data.data, data.target.toarray(), data.sample_id


def load_validation_data(path_to_ids):
    data = fetch_rcv1(subset='test')
    ids = pd.read_csv(path_to_ids, names=['id'], dtype=np.int32)
    mask = np.isin(data.sample_id, ids['id'])
    validation_data = data.data[mask]
    validation_target = data.target[mask].toarray()
    validation_ids = data.sample_id[mask]
    return validation_data, validation_target, validation_ids


class CS5304BaseClassifier(object):
    def __init__(self):
        pass

    def train(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


from sklearn.neighbors import KNeighborsClassifier
class CS5304KNNClassifier:
    
    def __init__(self, n_neighbors=3):
        super(CS5304KNNClassifier, self).__init__()
        self.n_neighbors = n_neighbors
    
    def train(self, X_train, y_train):
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X_train, y_train) 
        
    def predict(self, X_test):
        if self.model:
            return self.model.predict(X_test)
        else:
            raise "No model has been trained yet"
    
    def get_score(self, X_test, y_test):
        if self.model:
            return self.model.score(X_test, y_test)
        else:
            raise "No model has been trained yet"


from sklearn.naive_bayes import BernoulliNB

class CS5304NBClassifier(CS5304BaseClassifier):
    
    def __init__(self):
        super(CS5304NBClassifier, self).__init__()
    
    def train(self, X_train, y_train):
        self.model = BernoulliNB().fit(X_train, y_train)
    
    def predict(self, X_test):
        if self.model:
            return self.model.predict(X_test)
        else: 
            raise "No model has been trained yet"   
    
    def get_score(self, X_test, y_test):

        if self.model:
            return self.model.score(X_test, y_test)
        else:
            raise "No model has been trained yet"
        


class CS5304KMeansClassifier(CS5304BaseClassifier):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
from sklearn.cluster import KMeans
import scipy
class CS5304KMeansClassifier():
    
    def __init__(self, n_clusters = 2):
        self.n_clusters = n_clusters
    
    def train(self, X, y):
        init_centroids = self.get_init_centroid(X,y)
        self.model = KMeans(n_clusters=self.n_clusters, init=init_centroids, n_init=1).fit(X)
    
    def predict(self, X):
        if self.model: 
            return self.model.predict(X)
        else:
            raise "No model has been trained yet"
    
    def get_init_centroid(self, X, y):
        true_set = [i for i, j in enumerate(y) if j == 1]
        false_set = [i for i, j in enumerate(y) if j == 0]
        true_cluster = X[true_set]
        false_cluster = X[false_set]
        true_centroid = X[true_set].mean(axis = 0)
        false_centroid = X[false_set].mean(axis = 0)
        return np.concatenate((false_centroid, true_centroid), axis = 0)
    
    def get_score(self, X, y):
        if self.model:
            n_samples = X.shape[0]
            y_hat = np.array(self.model.predict(X)).flatten()
            y_true_label = np.array(y).flatten()
            count = len([i for i in range(n_samples) if y_hat[i] == y_true_label[i]])
            return count / float(n_samples)
        else:
            raise "No model has been trained yet"
    
    def get_final_centroid(self):
        return self.model.cluster_centers_
    


if __name__ == '__main__':

    # This is an example of loading the training and validation data. You may use this snippet
    # when completing the exercises for the assignment.

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_labels", default="labels.txt")
    parser.add_argument("--path_to_ids", default="validation.txt")
    options = parser.parse_args()

    labels = load_labels(options.path_to_labels)
    train_data, train_target, _ = load_training_data()
    eval_data, eval_target, _ = load_validation_data(options.path_to_ids)
