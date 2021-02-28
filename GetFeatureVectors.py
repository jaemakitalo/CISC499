import GetClusterLabels as gcl

#cluster the features of each superclass within the dataset
    #loops through the dataset
#calls getClusterLabels within the loop

#data = reduced data instance
#y = superclass label
def getFeatureVectors(data, y):
    featureVectors = []
    for i in range(len(y)):
        output = gcl.getClusterLabels(data[i], y[i])
        featureVectors.append (output)
    return featureVectors
