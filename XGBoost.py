from xgboost import XGBClassifier
import numpy as np


def train(labels, features):

    clf = XGBClassifier(class_weight='balanced', random_state=0)

    clf = clf.fit(features, labels)

    return clf


def run(clf, features):
    predictions = clf.predict(features)
    return predictions


def run_prob(clf, features):
    predictions = clf.predict_proba(features)