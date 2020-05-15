from sklearn.svm import SVC


def train(labels, features, *args):
    clf = SVC(C=1, kernel=args[0], gamma='scale', class_weight='balanced', probability=True)
    clf.fit(features, labels)

    return clf


def run(clf, features):

    predictions = clf.predict(features)

    return predictions


def run_prob(clf, features):

    predictions = clf.predict_proba(features)

    return predictions