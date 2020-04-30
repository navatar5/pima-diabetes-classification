from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt



def train(labels, features, *args):
    # args[0] will be the ccp_alpha parameter if used
    if len(args) < 1:
        clf = RandomForestClassifier(class_weight="balanced", max_depth=7, min_samples_split=5, random_state=0)
    else:
        clf = RandomForestClassifier(class_weight="balanced", random_state=0, ccp_alpha=args[0])

    clf = clf.fit(features, labels)

    return clf


def run(clf, features):
    predictions = clf.predict(features)
    return predictions


def run_prob(clf, features):
    predictions = clf.predict_proba(features)
    return predictions


def show_feature_importances(clf, features):
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(features.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(features.shape[1]), indices)
    plt.xlim([-1, features.shape[1]])
    plt.show()

