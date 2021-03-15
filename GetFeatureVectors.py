import GetClusterLabels as gcl
import GetData as gd
import pandas as pd

'''
The featureVectors will look like:
    {superclassLabel1: [[...], [...], ... , [...]], superclassLabel2: [[...], [...], ... , [...]]}
'''

#cluster the features of each superclass within the dataset
    #loops through the dataset
#calls getClusterLabels within the loop

#data = reduced data instance
def getFeatureVectors(datasetName):
    data = gd.getData(datasetName)  #Modify this line after we have the dimensionality reducer
    featureVectors = {}
    for i in range(len(data)):
        #data[i][:-1] -- data without superclass labels, data[i][-1] -- superclass labels
        featureVectors = gcl.getClusterLabels(pd.DataFrame(data.loc[i][:-1]), data.loc[i][-1], featureVectors)
    return featureVectors
