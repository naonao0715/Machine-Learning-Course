import random
import numpy as np
import pdb
import cluster
import sklearn

from sklearn.datasets import make_blobs
random.seed(10)
class KMeans(cluster.cluster):
    def __init__(self, k = 5, max_iterations = 100):
        self.k = k
        self.max_iterations = max_iterations
    
    def fit(self, X):
        n = len(X)
        d = len(X[0])
        X_np = np.array(X)
        clusters = np.array([], np.float64).reshape(0, d)
        for i in range(self.k):
            idx = random.randint(0, n)
            clusters = np.vstack([clusters, X_np[idx]])
        for iter in range(self.max_iterations):
            dist_clusters = np.array([], np.float64).reshape(n, 0)
            for i in range(self.k):
                diff = np.sum((X_np - clusters[i])**2, axis = 1)
                dist_clusters = np.column_stack([dist_clusters, diff])
        
            assigned_clusters = np.argmin(dist_clusters, axis = 1)

            for i in range(self.k):
                clusters[i] = np.mean(X_np[(assigned_clusters == i)], axis = 0)
        self.clusters = clusters
        return assigned_clusters, clusters
    
    def inertia(self, X):
        ## compute inertia
        n = len(X)
        d = len(X[0])
        X_np = np.array(X)
        dist_clusters = np.array([], np.float64).reshape(n, 0)
        for i in range(self.k):
            diff = np.sum((X_np - self.clusters[i])**2, axis = 1)
            dist_clusters = np.column_stack([dist_clusters, diff])

        assigned_clusters = np.argmin(dist_clusters, axis = 1)

        out = 0.
        for i in range(self.k):
            delta = X_np[(assigned_clusters == i)] - self.clusters[i]
            out += np.sum(delta**2) 
        return out
                    


    def fit_extended(self, X, balanced = False):
        if not balanced:
            return self.fit(X)
        n = len(X)
        d = len(X[0])
        X_np = np.array(X)
        clusters = np.array([], np.float64).reshape(0, d)
        for i in range(self.k):
            idx = random.randint(0, n)
            clusters = np.vstack([clusters, X_np[idx]])
        
        ## assign equal number of instance to each cluster based on the minimum distance. 
        ## if current cluster is full, assign to the next one with minimum distance
        dist_clusters = np.array([], np.float64).reshape(n, 0)
        for i in range(self.k):
            diff = np.sum((X_np - clusters[i])**2, axis = 1)
            dist_clusters = np.column_stack([dist_clusters, diff])
        sort_dist = []
        for i in range(n):
            for j in range(self.k):
                sort_dist.append((dist_clusters[i][j], i, j))
        
        sort_dist = sorted(sort_dist, key = lambda x: x[0])
        
        remain_cluster = np.full(self.k, n // self.k, dtype = int)
        assigned_clusters = np.full(n, -1, dtype = int)
        for i in range(n % self.k):
            remain_cluster[i] += 1
        
        for tup in sort_dist:
            if assigned_clusters[tup[1]] != -1:
                continue
            if remain_cluster[tup[2]] == 0:
                continue
            remain_cluster[tup[2]] -= 1
            assigned_clusters[tup[1]] = tup[2]
        
        
        for iter in range(self.max_iterations):
            ## recompute the cluster mean
            for i in range(self.k):
                clusters[i] = np.mean(X_np[(assigned_clusters == i)], axis = 0)

            ## compute delta between current assignment and other assignment
            dist_clusters = np.array([], np.float64).reshape(n, 0)
            for i in range(self.k):
                diff = np.sum((X_np - clusters[i])**2, axis = 1)
                dist_clusters = np.column_stack([dist_clusters, diff])
            sort_dist = []
            for i in range(n):
                for j in range(self.k):
                    sort_dist.append((dist_clusters[i][j] - dist_clusters[i][assigned_clusters[i]], i, j))
            sort_dist =sorted(sort_dist, key = lambda x: x[0])

            ## if other assignment is better than current assignment, 
            ## swap it with other assignment if there is candidate from other candidate
            ## otherwise put it into swap list
            is_sample_swapped = np.full(n, False, dtype=bool)
            remain_cluster = np.full(self.k, n // self.k, dtype = int)

            swap_list = {k: [] for k in range(self.k)}
            for tup in sort_dist:
            
                if tup[0] >= 0:
                    break
                idx = tup[1]
                if is_sample_swapped[idx] == True:
                    continue
                # try to swap with candidates in other cluster
                current_cluster = assigned_clusters[idx] 
                target_cluster = tup[2] 

                candidates = np.where(assigned_clusters == target_cluster)[0]
                
                
                min_ind = -1
                min_dist = 0
                for cand in candidates:
                    if is_sample_swapped[cand]:
                        continue
                    
                    tmp = dist_clusters[cand, current_cluster] - dist_clusters[cand, target_cluster]
                    if tmp > 0:
                        continue

                    if min_ind == -1 or tmp < min_dist:
                        min_ind = cand
                        min_dist = tmp
                
                if min_ind != -1:
                    assigned_clusters[idx] = target_cluster
                    assigned_clusters[min_ind] = current_cluster
                    is_sample_swapped[idx] = True
                    is_sample_swapped[min_ind] = True

        
        for i in range(self.k):
            clusters[i] = np.mean(X_np[(assigned_clusters == i)], axis = 0)
        self.clusters = clusters
        return assigned_clusters, clusters
                    


if __name__ == '__main__':

    X, cluster_assignments = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=0)

    kmean = KMeans(4)
    clusters_predicted, clusters_center = kmean.fit(X)
    clusters_predicted_extended, clusters_cente_extended = kmean.fit_extended(X, True)
    
    np.savetxt('kmeans_default.txt', clusters_predicted, delimiter=',', fmt='%i') 
    np.savetxt('kmeans_extended.txt', clusters_predicted_extended, delimiter=',', fmt='%i') 