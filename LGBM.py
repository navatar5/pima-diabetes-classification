from lightgbm import LGBMClassifier
import numpy as np


def train(labels, features):

    clf = LGBMClassifier(n_estimators=100, class_weight='balanced', max_bin=100, max_depth=5, num_leaves=20, random_state=0, learning_rate=0.003, reg_alpha=0.1, reg_lambda=0.01, subsample_freq=1, subsample=0.8, colsample_bytree=0.7)

    clf = clf.fit(features, labels)

    return clf


def run(clf, features):
    predictions = clf.predict(features)
    return predictions


def run_prob(clf, features):
    predictions = clf.predict_proba(features)
    return predictions
