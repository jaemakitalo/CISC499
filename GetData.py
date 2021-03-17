'''
Superclass labels are represented as targets in the PMLB datasets
Treat the data as a 2D matrix
Use the same Python list manipulation to obtain specific values from the data

Examples:
    data = getData("analcatdata_aids")
    print(data.loc[0])  #Read the first row of the data
    print(data.loc[0][:-1]) #Read the first row of the data without the superclass label -- method 1
    print(pd.DataFrame(data.loc[i][:-1]))   #Read the first row of the data without the superclass label -- method 2
'''

import pandas as pd
import os

#Get data of a specific dataset
def getData(datasetName):
    datasetPath = os.path.join("datasets", datasetName + ".tsv")
    try:
        data = pd.read_csv(datasetPath, sep = "\t")
        headers = list(data.columns[:-1])
    except:
        raise FileNotFoundError("Invalid dataset name")
    return data, headers 
