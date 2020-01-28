import numpy as np
import datatools
import NaiveBayes
import SVM
import DTscratch
# Choose between leave one out CV or nFold CV


def leave_one_out(dataset, labels, algo):

    data_len = len(labels)
    # randomize the dataset
    rand_inds = np.random.permutation(data_len)

    predictions = np.zeros(data_len)

    for i in range(data_len):
        test_ind = rand_inds[i]  # choose 1 test sample at random
        train_inds = np.concatenate((rand_inds[0:i], rand_inds[i+1:]))

    # train on the rest of the data
        train_features = dataset[train_inds, :]
        train_labels = labels[train_inds]

        test_feature = np.reshape(dataset[test_ind, :], [1, len(dataset[test_ind, :])])

        trained_clf = algo.train(train_labels, train_features)  # Check algo parameters

        predictions[test_ind] = datatools.classify(algo.run(trained_clf, test_feature))

        if i % 100 == 0:
            print('%d samples run out of %d' % (i, data_len))
    return predictions


def n_fold(dataset, labels, algo, *args, n_folds):
    predictions = np.full(len(labels), np.nan)
    num_classes = np.max(labels)+1
    # separate by label
    class_inds = {}

    for i in range(num_classes):
        class_inds[i] = np.argwhere(labels == i)
        np.random.shuffle(class_inds[i])  # randomize the indices
    test_inds = {new_list: [] for new_list in range(n_folds)}
    train_inds = {new_list: [] for new_list in range(n_folds)}
    # get number of samples per fold for each label
    # go through each class and get the indices for each fold
    for j in range(num_classes):
        num_samples_fold = int(np.round(len(class_inds[j])/n_folds))
        start = 0
        # add the indices for each fold for each class
        for i in range(n_folds):
            test_inds[i] = np.append(test_inds[i], class_inds[j][start:start+num_samples_fold, :]).astype(int)
            train_inds[i] = np.append(train_inds[i], np.concatenate((class_inds[j][0:start], class_inds[j][start+num_samples_fold:]))).astype(int)
            start = start + num_samples_fold

    # make the classes not stacked on one other
    for i in range(n_folds):
        np.random.shuffle(test_inds[i])
        np.random.shuffle(train_inds[i])

    for i in range(n_folds):
        print('Fold %d out of %d' % (i+1, n_folds))

        train_features = dataset[train_inds[i], :]
        train_labels = labels[train_inds[i]]

        test_features = dataset[test_inds[i], :]

        if algo == NaiveBayes:
            trained_clf = algo.train(train_labels, train_features)
            predictions[test_inds[i]] = datatools.classify(algo.run(trained_clf, test_features))

        if algo == SVM:
            kernel = args[0]
            trained_clf = algo.train(train_labels, train_features, kernel)
            predictions[test_inds[i]] = algo.run(trained_clf, test_features)

        if algo == DTscratch:
            max_depth = args[0]
            min_samples_split = args[1]
            trained_clf = algo.train(train_labels, train_features, max_depth, min_samples_split)
            predictions[test_inds[i]] = datatools.classify(algo.run(trained_clf, test_features))

    return predictions


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    filename = "pima-indians-diabetes.data.csv"
    labels, features = datatools.loadDataset(filename)

    CV_preds = n_fold(features, labels, DTscratch, 10, 5, n_folds=10)
    print("Naive Bayes leave one out CV accuracy", datatools.accuracy(labels, CV_preds))
