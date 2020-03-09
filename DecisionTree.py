from sklearn import tree
import matplotlib.pyplot as plt
import graphviz


def train(*args):
    if len(args) < 3:
        labels = args[0]
        features = args[1]
        feature_names = None
        class_names = None
    else:
        labels = args[0]
        features = args[1]
        feature_names = args[2]
        class_names = args[3]

    clf = tree.DecisionTreeClassifier(class_weight="balanced", max_depth=10, min_samples_split=5)
    clf = clf.fit(features, labels)
    # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=class_names,
    #                                 filled=True, rounded=True, special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("tree")
    return clf


def train_ccp(labels, features, ccp_alpha):
    clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=0, ccp_alpha=ccp_alpha)
    clf = clf.fit(features, labels)

    return clf


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