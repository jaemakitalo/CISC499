#predicts superclass labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import GetData as GD

#agaricus-lepiota works well (likely due to large number of samples)
'''
def gettingData ():
    data = GD.getData("analcatdata_aids")
    return data

#adds header labels to the data
def addHeaders (data, headers):
    data.columns = headers
    return data
'''
#divides data into training and testing data
def divideData (dataset, train_percentage, feature_headers, target_header):
    #test_size: proportion of the dataset to include in the test split
    #train_size: not given, so will be complement of test_size
    #random_state: Controls the shuffling applied to the data before applying the split
    xTrain, xTest, yTrain, yTest = train_test_split(dataset[feature_headers], dataset[target_header], test_size=0.33, random_state=0)
    return xTrain, xTest, yTrain, yTest

#trains random forest classifier with features and target data
def rfc (features, target):
    classify = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classify.fit(features, target)
    return classify
    #criterion = gini or entropy
    #distribution of features, choose top 20%

def main ():
    data, headers = gettingData()
    dataset = addHeaders (data, headers)

    xTrain, xTest, yTrain, yTest = divideData(dataset, 0.7, headers[1:-1], headers[-1])

    #creating random forest classifier instance
    trainedModel = rfc (xTrain, yTrain)

    #select important features
    sfm = SelectFromModel(trainedModel, threshold=1e-2)
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
    trainedModelImportant = rfc (impXTrain, yTrain)

    #prints list of all the parameters for rfc
    #print ("Trained model: ", trainedModel)

    #applying full featured classifier to test data
    predictions = trainedModel.predict(xTest)

    print ("Initial training:")
    #taking some examples to see actual vs. predicted values
    for j in range(0, 5):
        print ("Actual: ", list(yTest)[j], "and Predicted: ", predictions[j])

    #accuract tests for the initial classifier
    print ("Train Accuracy: ", accuracy_score(yTrain, trainedModel.predict(xTrain)))
    print ("Test Accuracy: ", accuracy_score(yTest, predictions))
    print ("Confusion Matrix: \n", confusion_matrix(yTest, predictions))
    print ("Classification Report: \n", classification_report(yTest, trainedModel.predict(xTest)))

    #applying full featured classifier to the selected test data
    impPredictions = trainedModelImportant.predict(impXTest)
    print ("Training after selection:")

    for k in range(0, 5):
        print ("Actual: ", list(yTest)[k], "and Predicted: ", impPredictions[k])

    print ("Train Accuracy: ", accuracy_score(yTrain, trainedModelImportant.predict(impXTrain)))
    print ("Test Accuracy: ", accuracy_score(yTest, impPredictions))
    print ("Confusion Matrix: \n", confusion_matrix(yTest, impPredictions))
    print ("Classification Report: \n", classification_report(yTest, trainedModelImportant.predict(impXTest)))

#%main()