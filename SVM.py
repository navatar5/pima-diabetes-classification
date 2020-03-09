from sklearn.svm import SVC


def train(labels, features, kernel):
    clf = SVC(C=1, kernel=kernel, gamma='scale', class_weight='balanced')
    clf.fit(features, labels)

    return clf


def run(clf, features):

    predictions = clf.predict(features)

    return predictions

