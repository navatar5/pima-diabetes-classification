import numpy as np
from math import *


def train(labels, features):
    # this is the training function that generates the feature summaries
    # organize features by label in a dictionary
    summary = {}
    for i in range(np.max(labels) + 1):
        class_inds = np.argwhere(labels == i)
        summary[i] = features[class_inds]

    # organize the mean and std deviation of each feature with the class label
    for i in range(len(summary)):
        mean = np.squeeze(np.mean(summary[i], axis=0))  # calculate mean along the columns
        std_dev = np.squeeze(np.std(summary[i], axis=0))  # calculate the std deviation along the columns

        summary[i] = {'mean': mean, 'std_dev': std_dev}

    return summary


def normPDF(sample, mean, std_dev):
    # samples from a gaussian PDF
    prob = (1/sqrt(2*pi*pow(std_dev, 2)))*exp(-(pow((sample - mean), 2))/(2*pow(std_dev, 2)))

    return prob


def calcProbs(summary, features):
    # given a new set of features, sample gaussian distributions for each feature given the class
    # P(class_i|data) = P(class_i)*P(class_i|f_1)*P(class_i|f_2)....
    # or log(P(class_i|data)) = log(P(class_i)) + log(P(class_i|f_1)) + log(P(class_i|f_2)) ...

    prob = np.zeros([len(summary)])
    for classVal in range(len(summary)):
        for i in range(len(features)):
            prob[classVal] = prob[classVal] * normPDF(features[i], summary[classVal]['mean'][i], summary[classVal]['std_dev'][
                i])  # prior plus conditionals

    # prob = np.exp(logProb)
    prob = prob/np.sum(prob)  # normalizing
    return prob


def run(summary, data):
    # returns class probabilites of new data
    probs = np.zeros([data.shape[0], 2])
    for i in range(data.shape[0]):
        probs[i, :] = calcProbs(summary, data[i, :])

    return probs


if __name__ == "__main__":
    filename = "pima-indians-diabetes.data.csv"
    labels, features = datatools.loadDataset(filename)
    split = 0.7

    train_labels, test_labels, train_features, test_features = datatools.splitData(labels, features, split)
    dataSummary = summarizeData(train_labels, train_features)
    probabilities = predictProb(dataSummary, test_features)
    predictions = classify(probabilities)

    test_acc = accuracy(test_labels,predictions)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    print(test_acc)





