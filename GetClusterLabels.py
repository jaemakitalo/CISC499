
#Get superclass labels from the dataset
#vi (reduced data instance), yi (superclass label)
def getClusterLabels (vi, yi, zi):
    if yi in zi.keys():
        valueVector = zi.get(yi)
        valueVector.append([vi])
    else:
        zi[yi] = [[vi]]
    return zi
	#zi - a data structure containing all the data under the superclass label yi
