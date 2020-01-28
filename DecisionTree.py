from sklearn import tree
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
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=class_names,
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("tree")
    return clf


def run(clf, features):
    predictions = clf.predict(features)
    return predictions