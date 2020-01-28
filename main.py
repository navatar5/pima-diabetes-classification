import NaiveBayes
import SVM
import DecisionTree
import DTscratch
import datatools
import numpy as np
import pickle

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    filename = "pima-indians-diabetes.data.csv"
    labels, features = datatools.loadDataset(filename)
    feature_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI","DiPedFunc", "Age"]
    class_names = ["No Diabetes", "Diabetes"]
    split = 0.7

    train_labels, test_labels, train_features, test_features = datatools.splitData(labels, features, split)
    dataSummary = NaiveBayes.train(train_labels, train_features)
    NB_testProbs = NaiveBayes.run(dataSummary, test_features)
    NB_trainProbs = NaiveBayes.run(dataSummary, train_features)

    NB_trainPreds = datatools.classify(NB_trainProbs)
    NB_testPreds = datatools.classify(NB_testProbs)

    NBtrain_acc = datatools.accuracy(train_labels, NB_trainPreds)
    NBtest_acc = datatools.accuracy(test_labels, NB_testPreds)

    print("Naive Bayes Train Accuracy: ", NBtrain_acc)
    print("Naive Bayes Test Accuracy: ", NBtest_acc)

    trainedSVM = SVM.train(train_labels, train_features, 'linear')
    SVM_train_predictions = SVM.run(trainedSVM, train_features)
    SVM_test_predictions = SVM.run(trainedSVM, test_features)
    SVMtrain_acc = datatools.accuracy(train_labels, SVM_train_predictions)
    SVMtest_acc = datatools.accuracy(test_labels, SVM_test_predictions)
    print("SVM Train Accuracy: ", SVMtrain_acc)
    print("SVM Test Accuracy: ", SVMtest_acc)

    trainedTree = DecisionTree.train(train_labels, train_features, feature_names, class_names)
    DT_train_predictions = DecisionTree.run(trainedTree, train_features)
    DT_test_predictions = DecisionTree.run(trainedTree, test_features)
    DTtrain_acc = datatools.accuracy(train_labels, DT_train_predictions)
    DTtest_acc = datatools.accuracy(test_labels, DT_test_predictions)
    print("DT Train Accuracy: ", DTtrain_acc)
    print("DT Test Accuracy: ", DTtest_acc)

    trainedTree_scratch = DTscratch.train(train_labels, train_features, 10, 5)
    DTscratch_train_predictions = datatools.classify(DTscratch.run(trainedTree_scratch, train_features))
    DTscratch_test_predictions = datatools.classify(DTscratch.run(trainedTree_scratch, test_features))
    DTscratchtrain_acc = datatools.accuracy(train_labels, DTscratch_train_predictions)
    DTscratchtest_acc = datatools.accuracy(test_labels, DTscratch_test_predictions)
    print("DTscratch Train Accuracy: ", DTscratchtrain_acc)
    print("DTscratch Test Accuracy: ", DTscratchtest_acc)



