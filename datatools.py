# Contains function for loading and splitting data
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

def loadDataset(filename):

    if filename[-3:] == 'csv':
        lines = csv.reader(open(filename, "r"))
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        dataset = np.asarray(dataset)
        labels = dataset[:, -1].astype(int)
        features = dataset[:, 0:-1]

        return labels, features


def splitData(labels, features, split):
    # splits data into training and test sets based on a split fraction
    # create a dictionary of the indices of each class
    class_inds = {}
    for i in range(np.max(labels)+1):
        class_inds[i] = np.argwhere(labels == i)
        np.random.shuffle(class_inds[i])  # randomize the indices

    train_inds = []
    test_inds = []

    for i in range(np.max(labels) + 1):
        train_inds = np.append(train_inds, class_inds[i][0:round(split*len(class_inds[i]))]).astype(int)
        test_inds = np.append(test_inds, class_inds[i][round(split*len(class_inds[i])):]).astype(int)

    # randomize the train and test indices so that the classes are not stacked on one other
    np.random.shuffle(train_inds)
    np.random.shuffle(test_inds)

    train_labels = labels[train_inds]
    test_labels = labels[test_inds]

    train_features = features[train_inds, :]
    test_features = features[test_inds, :]

    return train_labels, test_labels, train_features, test_features


def accuracy(labels, predictions):
    # returns accuracy of predictions
    acc = np.sum(predictions == labels)/len(labels) * 100
    return acc


def classify(probs):
    # returns the classifications given class probabilities
    preds = np.argmax(probs, 1)
    return preds


def binary_metrics(labels, predictions, *args):
    conf = confusion_matrix(labels, predictions)
    fp_rate = conf[0, 1] / (conf[0, 1] + conf[0, 0])
    tp_rate = conf[1, 1] / (conf[1, 1] + conf[1, 0])

    if args[0] == 'tp':
        return tp_rate

    if args[0] == 'fp:':
        return fp_rate

    if args[0] == 'acc':
        return accuracy(labels, predictions)


