from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import graphviz


def train(labels, features, *args):
    # args[0] will be the ccp_alpha parameter if used
    if len(args) < 1:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", max_depth=10, min_samples_split=5)
        clf = clf.fit(features, labels)
    else:
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=0, ccp_alpha=args[0])
        clf = clf.fit(features, labels)

    return clf



def show_feature_importances(clf, features):
    importances = clf.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(features.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features.shape[1]), importances[indices],
            color="r", align="center")
    plt.xticks(range(features.shape[1]), indices)
    plt.xlim([-1, features.shape[1]])
    plt.show()


def get_ccp_alphas(train_labels, train_features):
    clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=0)  # set random state for repeatability
    path = clf.cost_complexity_pruning_path(train_features, train_labels)  # get the path
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")

    return ccp_alphas


def run(clf, features):
    predictions = clf.predict(features)
    return predictions

def run_prob(clf, features):
    predictions = clf.predict_proba(features)
    return predictions