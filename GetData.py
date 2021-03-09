'''
Superclass labels are represented as targets in the PMLB datasets
Treat the data as a 2D matrix
Use the same Python list manipulation to obtain specific values from the data

Examples:
    data = getData("analcatdata_aids")
    print(data.loc[0])  #Read the first row of the data
    print(data.loc[0][:-1]) #Read the first row of the data without the superclass label
'''

import pandas as pd
import os
import requests

#Get data of a specific dataset
def getData(datasetName):
    url = "https://github.com/EpistasisLab/pmlb/raw/master/datasets"
    extension = ".tsv.gz"
    datasetURL = getURL(url, datasetName, extension)
    data = pd.read_csv(datasetURL, sep = "\t", compression = "gzip")
    return data

#Get the URL of the dataset from the PMLB Github
def getURL(url, datasetName, extension):
    datasetURL = os.path.join(url, datasetName, datasetName + extension)
    requestURL = requests.get(datasetURL)
    #Check if the provided datasetName is valid
    if requestURL.status_code != 200:
        raise ValueError('The provided dataset name is invalid')
    return datasetURL
