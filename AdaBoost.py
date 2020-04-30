from sklearn.ensemble import AdaBoostClassifier
import numpy as np


def train(labels, features):

    clf = AdaBoostClassifier(n_estimators=1000, random_state=0)

    clf = clf.fit(features, labels)

    return clf


def run(clf, features):
    predictions = clf.predict(features)
    return predictions


def run_prob(clf, features):
    predictions = clf.predict_proba(features)
    return predictions
