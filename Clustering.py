from sklearn.cluster import KMeans as clustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#puts the data into clusters

#n = cluster number - do 2-10, use the one that gives most
#success

# Using k-means algorithm
#https://scikit-learn.org/stable/modules/clustering.html#k-means
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
# Helpful Article: https://heartbeat.fritz.ai/k-means-clustering-using-sklearn-and-python-4a054d67b187



def cluster (dataset):
    '''
    Note: got the same number of clusters using 
    PCA both before and after the iterative clustering
    '''
    
    # pass through data just under chosen headers
    print("Clustering the data...")
    # Clusters the data and visualizes best number of clusters using the elbow method
    Error =[]
    for i in range(2,11): 
        kmeans = clustering(n_clusters=i).fit(dataset)
        Error.append(kmeans.inertia_)
    plt.plot(range(2, 11), Error)
    plt.title('Elbow method')
    plt.xlabel('No of clusters')
    plt.ylabel('Error')
    plt.show()
    
    # allows user to input the best number of clusters
    k = int(input("How many clusters for k-means clustering: "))
    
    #reduces the data using PCA to graph data
    pca = PCA(2)
    data = pca.fit_transform(dataset)
    data.shape

    # clusters into k clusters inputted by user
    kcluster = clustering(n_clusters=k)
    y_kcluster= kcluster.fit_predict(data)
    print(" Cluster Labels: ")
    print(y_kcluster)
    print("Cluster Centers: ")
    print(kcluster.cluster_centers_)
    # visualizes data
    plt.title('K-MEANS Clustering')
    plt.scatter(data[:,0], data[:,1], c= y_kcluster, cmap='rainbow')
    plt.show()
    
    