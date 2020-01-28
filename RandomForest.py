from sklearn.ensemble import RandomForestClassifier


def train(labels, features):
    clf = RandomForestClassifier(class_weight="balanced", max_depth=10, min_samples_split=5)
    clf = clf.fit(features, labels)

    return clf


def run(clf, features):
    predictions = clf.predict(features)
    return predictions

