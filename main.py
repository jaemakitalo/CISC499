import GetData as GD
import Classifier as CLA
import Clustering as CLU
from sklearn.cluster import KMeans as clustering

def gettingData ():
    data = GD.getData("analcatdata_aids")
    return data

#adds header labels to the data
def addHeaders (data, headers):
    data.columns = headers
    return data

def data_classification():
    data, headers = gettingData()
    dataset = addHeaders (data, headers)
    print(dataset)

    xTrain, xTest, yTrain, yTest = CLA.divideData(dataset, 0.7, headers[1:-1], headers[-1])

    #creating random forest classifier instance
    trainedModel = CLA.rfc (xTrain, yTrain)

    #select important features
    # should modify threshold to be a percentage or ratio, not set number
    # can look at distribution of features for the dataset
    sfm = CLA.SelectFromModel(trainedModel, threshold=0.20)
    sfm.fit(xTrain, yTrain)

    #to see which features were selected
    important = []
    for i in sfm.get_support(indices = True):
        important.append(headers[i])
    print ("selected features:", important)

    #transforming the data to create a new dataset with only the important features
    impXTrain = sfm.transform(xTrain)
    impXTest = sfm.transform(xTest)

    #creating new random forest classifier instance
    trainedModelImportant = CLA.rfc (impXTrain, yTrain)

    #prints list of all the parameters for rfc
    #print ("Trained model: ", trainedModel)

    #applying full featured classifier to test data
    predictions = trainedModel.predict(xTest)
    
    print ("Initial training:")
    #taking some examples to see actual vs. predicted values
    for j in range(0, 5):
        print ("Actual: ", list(yTest)[j], "and Predicted: ", predictions[j])

    #accuract tests for the initial classifier
    print ("Train Accuracy: ", CLA.accuracy_score(yTrain, trainedModel.predict(xTrain)))
    print ("Test Accuracy: ", CLA.accuracy_score(yTest, predictions))
    print ("Confusion Matrix: \n", CLA.confusion_matrix(yTest, predictions))
    print ("Classification Report: \n", CLA.classification_report(yTest, trainedModel.predict(xTest)))
  
    #applying full featured classifier to the selected test data
    impPredictions = trainedModelImportant.predict(impXTest)

    print ("Training after selection:")

    for k in range(0, 5):
        print ("Actual: ", list(yTest)[k], "and Predicted: ", impPredictions[k])

    print ("Train Accuracy: ", CLA.accuracy_score(yTrain, trainedModelImportant.predict(impXTrain)))
    print ("Test Accuracy: ", CLA.accuracy_score(yTest, impPredictions))
    print ("Confusion Matrix: \n", CLA.confusion_matrix(yTest, impPredictions))
    print ("Classification Report: \n", CLA.classification_report(yTest, trainedModelImportant.predict(impXTest)))

    return dataset, important

if __name__ == "__main__":
    dataset, ft = data_classification()
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