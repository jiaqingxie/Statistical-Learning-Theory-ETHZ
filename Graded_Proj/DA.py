import sklearn as skl
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np
from treelib import Tree
import matplotlib.pyplot as plt
scaler=MinMaxScaler()


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, metric="euclidian"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.T_min = 1
        self.Kmax = n_clusters
        self.converg_rate = 0.95
        
        self.cluster_centers = None
        self.cluster_probs = None
 
        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()
        self.bifurcation_tree = Tree()

        # Not necessary, depends on your implementation
        self.bifurcation_tree_cut_idx = None

        # Add more parameters, if necessary. You can also modify any other
        # attributes defined above
    
 
    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """
        """Compute DAC for input vectors samples
        Preferred implementation of DAC as described in reference [1].
        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """

        np.random.seed(seed=self.random_state)
        K = 1              
        self.n_eff_clusters  += [K]
        prob_x = 1 / samples.shape[0] #uniform prior
        
        self.cluster_centers = np.mean(samples, axis=0).reshape(1,-1) 
        self.cluster_probs = np.ones((samples.shape[0], 1))

        self.prob_y = np.ones((1, 1)) 
        T = self.crit_temp(samples, 0, prob_x)+50
        self.temperatures += [T]
        self.distortions += [self._distortion(samples)]

        self.bifurcation_tree = Tree()
        self.bifurcation_tree.create_node(0, '0_0')
        self.tree_centers = [self.cluster_centers[0]]
        self.directions = [1]
        self.tree_d = [[] for x in range(self.n_clusters)]
        self.tree_temps = [[] for x in range(self.n_clusters)]
        self.tree_d[0] +=[np.linalg.norm(self.tree_centers[0] - self.cluster_centers[0])]
        self.tree_temps[0] +=[T]
        self.tree_offsets = [0]

        while T >= self.T_min:
            while True:
                last_cluster_centers = self.cluster_centers.copy()
                self.cluster_probs = self._calculate_cluster_probs(self.get_distance(samples, last_cluster_centers), T)
                p_y_x = self.cluster_probs 
                self.prob_y = np.array([])

                for k in range(K):
                    p_y_x_k = p_y_x[:,k]
                    p_y =np.mean(p_y_x_k)
                    y = np.sum(prob_x * np.expand_dims(p_y_x_k , axis=1) * samples , axis=0)/p_y  
                    self.prob_y =np.append(self.prob_y,p_y)
                    self.cluster_centers[k] = y

                distortion = self._distortion(samples) 
                if  np.linalg.norm(self.cluster_centers - last_cluster_centers)< 10e-5: 
                    break 

            T *= self.converg_rate 

            distortion = self._distortion(samples)
            self.temperatures += [T]
            self.distortions += [distortion]
            
            if K <= self.Kmax:
                for k in range(0, K):
                    T_crit = self.crit_temp(samples, k, prob_x) 
                    if T < T_crit:
                        if K < self.n_clusters:
                            K += 1

                        self.directions[k] = -1
                        self.directions += [1]

                        self.tree_offsets[k] = self.tree_d[k][-1]
                        self.tree_offsets += [self.tree_d[k][-1]]
                        self.tree_centers[k] = self.cluster_centers[k].copy()
                        self.tree_centers += [self.cluster_centers[k].copy()]
                        self.tree_d[k] += [self.tree_offsets[k] - np.linalg.norm(self.tree_centers[k] - self.cluster_centers[k])]
                        self.tree_d[K-1] += [self.tree_offsets[k] - np.linalg.norm(self.tree_centers[k] - self.cluster_centers[k])]
                        self.tree_temps[K-1] += [T]

                        if self.prob_y.shape[0] < self.n_clusters and self.cluster_centers.shape[0] < self.n_clusters:
                            new_cluster_centers = self.cluster_centers[k] + np.random.normal(0, 0.1, size = samples.shape[1])
                            self.cluster_centers = np.vstack((self.cluster_centers, new_cluster_centers))
                            self.prob_y = self.prob_y[:]
                            self.prob_y = np.append(self.prob_y, [self.prob_y[k]/2])
                            self.prob_y[k] /=2
                    else:
                        self.tree_d[k] += [self.tree_offsets[k] + self.directions[k] * np.linalg.norm(self.tree_centers[k] - self.cluster_centers[k])]
                    self.tree_temps[k] += [T]

            self.n_eff_clusters += [K]

        return self          
                                                    
    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        min_dist = np.amin(np.square(dist_mat), 1, keepdims=True)
        exponential = np.multiply(self.prob_y.T, np.exp((min_dist-np.square(dist_mat))/temperature))
        probs = exponential / np.sum(exponential, 1, keepdims=True)
        return probs
    

    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """
        if self.metric == "euclidian":
            dist_mat = np.transpose(np.array([np.square(np.linalg.norm(samples - clusters[k], axis=1)) for k in range(clusters.shape[0])]))
        return np.sqrt(dist_mat)
    

    def crit_temp(self, samples, k, prob_x):
        p_y =  self.prob_y[k]
        C_x_y = np.zeros((samples.shape[1],samples.shape[1]))
        x_y = samples - self.cluster_centers[k]
        for i in range(samples.shape[0]):
            x_y_i = x_y[i,:].reshape(1,-1)
            C_x_y += np.dot(prob_x, self.cluster_probs[i, k]) / p_y * x_y_i.T @ x_y_i #page 6 formula 18
        return 2 * np.amax(np.linalg.eigvals(C_x_y)) 
        
    def _distortion(self, samples):
        dist_mat = np.square(self.get_distance(samples, self.cluster_centers))
        distortion = np.sum(self.cluster_probs * dist_mat)/ samples.shape[0]
        return distortion

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting

        This is a pseudo-code showing how you may be using the tree
        information to make a bifurcation plot. Your implementation may be
        entire different or based on this code.
        """
        check_is_fitted(self, ["bifurcation_tree"])
        plt.figure(figsize=(8, 8))

        for k in range(self.n_clusters):
            plt.plot(self.tree_d[k], 1/np.array(self.tree_temps[k]), label=str(k))
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$ 1/T$')
        plt.title('Bifurcation Plot')
        plt.show()

    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()
