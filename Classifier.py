#predicts superclass labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import GetData as GD

def gettingData ():
    data = GD.getData ("analcatdata_aids")
    #print (data.head()) #to see some of the data fully

    return data


def divideData (dataset, train_percentage, feature_headers, target_header):
    #dividing data
    #test_size: proportion of the dataset to include in the test split
    #train_size: not given, so will be complement of test_size
    #random_state: Controls the shuffling applied to the data before applying the split
    xTrain, xTest, yTrain, yTest = train_test_split(data[feature_headers], data[target_header], test_size=0.2, random_state=0)

    return xTrain, xTest, yTrain, yTest

def rfc (features, target):
    classify = RandomForestClassifier(n_estimators=20, random_state=0)
    classify.fit(features, target)

    return classify


def main ():
    data = gettingData()
    # xTrain, xTest, yTrain, yTest = divideData(data, 0.7, headers[1:-1], headers[-1])
    # print "xTrain Shape :: ", xTrain.shape
    # print "xTest Shape :: ", xTest.shape
    # print "yTrain Shape :: ", yTrain.shape
    # print "yTest Shape :: ", yTest.shape
    #
    # trainedModel = rfc (xTrain, yTrain)
    # print ("Trained model: ", trainedModel)

main()
