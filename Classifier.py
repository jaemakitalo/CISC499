#predicts superclass labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import GetData as GD

#agaricus-lepiota works well (likely due to large number of samples)
def gettingData ():
    data = GD.getData("analcatdata_aids")
    return data

#adds header labels to the data
def addHeaders (data, headers):
    data.columns = headers
    return data

#divides data into training and testing data
def divideData (dataset, train_percentage, feature_headers, target_header):
    #test_size: proportion of the dataset to include in the test split
    #train_size: not given, so will be complement of test_size
    #random_state: Controls the shuffling applied to the data before applying the split
    xTrain, xTest, yTrain, yTest = train_test_split(dataset[feature_headers], dataset[target_header], test_size=0.33, random_state=0)
    return xTrain, xTest, yTrain, yTest

#trains random forest classifier with features and target data
def rfc (features, target):
    classify = RandomForestClassifier(n_estimators=100, random_state=0)
    classify.fit(features, target)
    return classify


def main ():
    data, headers = gettingData()
    headers.append ('target') #headers list from GetData does not the value 'target'

    dataset = addHeaders (data, headers)

    xTrain, xTest, yTrain, yTest = divideData(dataset, 0.7, headers[1:-1], headers[-1])
    #printing to see if data was divided correctly
    # print ("xTrain Shape : ", xTrain.shape)
    # print ("yTrain Shape : ", yTrain.shape)
    # print ("xTest Shape : ", xTest.shape)
    # print ("yTest Shape : ", yTest.shape)

    #creating random forest classifier instance
    trainedModel = rfc (xTrain, yTrain)
    #prints list of all the parameters for rfc
    print ("Trained model: ", trainedModel)
    predictions = trainedModel.predict(xTest)

    #taking some examples to see actual vs. predicted values
    for i in range(0, 5):
        print ("Actual: ", list(yTest)[i], "and Predicted: ", predictions[i])


    print ("Train Accuracy: ", accuracy_score(yTrain, trainedModel.predict(xTrain)))
    print ("Test Accuracy: ", accuracy_score(yTest, predictions))
    print ("Confusion Matrix: \n", confusion_matrix(yTest, predictions))
    print ("Classification Report: \n", classification_report(yTest, trainedModel.predict(xTest)))

main()
