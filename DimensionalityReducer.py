from sklearn.decomposition import PCA
import GetData as gd


def dimensionalityReducer(datasetName):
    data, headers = gd.getData(datasetName) 
