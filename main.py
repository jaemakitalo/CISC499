import Classifier as CLA
import Clustering as CLU
from sklearn.cluster import KMeans as clustering

if __name__ == "__main__":
    dataset = CLA.classifying()
    # pass through data just under chosen headers
    feat = []
    print("Clustering the data...")
    kmeans = clustering(n_clusters=4, random_state=0).fit(dataset)
    print("predicted labels: ")
    print(kmeans.labels_)
    print("Cluster centers: ")
    print(kmeans.cluster_centers_)

'''
figures:
    feature importance distribution
    clustering results
    plot classification performance
    testing accuracy

Think about the intermediate steps on how to get to the final result
Think about interesting observations of the final results:
    why do we have this result, take different perspective to make
    more figures to try and answer the why question

For poster, most of the info should be represented using figures and tables
(have some flow chats for the methodoly and 2-3 for the results)
Do not want entire results section by its own. Use subsections for
to represent the final results

Keep audience in mind - tailor our material/presentation to their background

'''
