from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt


def train(labels, features, *args):
    # args[0] will be the ccp_alpha parameter if used
    if len(args) < 1:
        clf = GradientBoostingClassifier(max_depth=7, min_samples_split=5, random_state=0)
    else:
        clf = GradientBoostingClassifier(random_state=0, ccp_alpha=args[0])

    clf = clf.fit(features, labels)

    return clf


def run(clf, features):
    predictions = clf.predict(features)
    return predictions


def run_prob(clf, features):
    predictions = clf.predict_proba(features)
    return predictions