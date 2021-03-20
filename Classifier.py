#predicts superclass labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import GetData as GD

def gettingData ():
    data = GD.getData("analcatdata_aids")
    return data

def addHeaders (data, headers):
    data.columns = headers
    return data

def datacsv (headers):
    dataset = pd.read_csv()

def divideData (dataset, train_percentage, feature_headers, target_header):
    #dividing data
    #test_size: proportion of the dataset to include in the test split
    #train_size: not given, so will be complement of test_size
    #random_state: Controls the shuffling applied to the data before applying the split
    xTrain, xTest, yTrain, yTest = train_test_split(dataset[feature_headers], dataset[target_header], test_size=0.2, random_state=0)

    return xTrain, xTest, yTrain, yTest

def rfc (features, target):
    classify = RandomForestClassifier(n_estimators=20, random_state=0)
    classify.fit(features, target)

    return classify


def main ():
    data, headers = gettingData()
    headers.append ('target')
    #print (data)
    #print (headers)

    dataset = addHeaders (data, headers)
    #print (dataset)


    xTrain, xTest, yTrain, yTest = divideData(dataset, 0.7, headers[1:-1], headers[-1])
    print ("xTrain Shape :: ", xTrain.shape)
    print ("yTrain Shape :: ", yTrain.shape)
    print ("xTest Shape :: ", xTest.shape)
    print ("yTest Shape :: ", yTest.shape)

    trainedModel = rfc (xTrain, yTrain)
    print ("Trained model: ", trainedModel)

main()
