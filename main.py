import Classifier as CLA
import Clustering as CLU

if __name__ == "__main__":
    dataset = CLA.classifying()
    CLU.cluster(dataset)

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
