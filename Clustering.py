#puts the data into clusters

#n = cluster number - do 2-10, use the one that gives most
#success

# Using k-means algorithm
#https://scikit-learn.org/stable/modules/clustering.html#k-means
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

from sklearn.cluster import KMeans as clustering

def clustering (data, n):
    #k-means
    '''
    Parameters:
    n-clusters (int) - default=8
        number of clusters & number of centroids to generate
    init - default=k-means++
        method for initialization: selects initial cluster centers in a smart way
    n_init (int) - default=10
        number of times the k-means algorithm will run with different centroid seeds.
        Results will be best output of this
    max_iter (int) - default=300
        number of iterations of k-means algorithm
    precompute_distances - default=auto
        do not precompute if n_samples*n_clusters > 12 milliom
    
    '''
    return clusters
    
