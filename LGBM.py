from lightgbm import LGBMClassifier
import numpy as np


def train(labels, features):

    clf = LGBMClassifier(random_state=0, n_estimators=100, class_weight='balanced', num_leaves=25, objective='binary', learning_rate=.05, max_bin=200, metric=['auc', 'binary_logloss'], reg_alpha=0.5, reg_lambda=0.5)

    clf = clf.fit(features, labels)

    return clf


def run(clf, features):
    predictions = clf.predict(features)
    return predictions


def run_prob(clf, features):
    predictions = clf.predict_proba(features)
    return predictions
