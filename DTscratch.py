import numpy as np


def gini_index(groups, labelGroups):
    # labels is a list of the lists of labels for each group
    # total number of samples

    allLabels = []
    for labelGroup in labelGroups:
        for i in range(len(labelGroup)):
            allLabels.append(labelGroup[i])
    allLabels = np.asarray(allLabels)
    numSamples = len(allLabels)
    classVals = range(int(np.max(allLabels)+1))
    gini = 0.0
    for i in range(len(groups)):
        groupSize = float(len(groups[i]))
        if groupSize == 0:
            continue
        groupLabels = labelGroups[i]

        # Gini = (1 - sum(p*p)) * (group_size/total_samples)

        # compute the term sum(p*p): propSum
        propSum = 0.0
        for classVal in classVals:
            # calculate proportion of each class in the group
            p = np.sum(groupLabels == classVal)/groupSize
            propSum = propSum + p*p

        # weight the group by its size
        gini = gini + ((1.0 - propSum) * (groupSize/numSamples))

    return gini


def split(feature, split_val, feature_sets):
    left_split_inds, right_split_inds = [], []
    for i in range(len(feature_sets)):
        if feature_sets[i][feature] < split_val:
            left_split_inds.append(i)
        else:
            right_split_inds.append(i)
    return left_split_inds, right_split_inds


def get_best_split(features, labels):
    best_feature, best_split_val, best_groups, best_group_labels = None, None, None, None
    best_gini = 1  # initialize to bad gini value
    # grid search to find what split results in best gini

    # start splitting based on feature
    for feature in range(len(features[0])):
        for i in range(len(features)):
            # iterate through each row of features
            left_group_inds, right_group_inds = split(feature, features[i][feature], features)
            left_group = features[left_group_inds]
            left_group_labels = labels[left_group_inds]
            right_group = features[right_group_inds]
            right_group_labels = labels[right_group_inds]

            groups = [left_group, right_group]
            group_labels = [left_group_labels, right_group_labels]

            gini = gini_index(groups, group_labels)
            # print('X%d < %.3f Gini=%.3f' % ((feature + 1), features[i][feature], gini))
            # update gini such that a lower gini becomes the new best gini
            if gini < best_gini:
                best_feature, best_split_val, best_gini, best_groups, best_group_labels = feature, features[i][feature], gini, groups, group_labels

    node = {'feature': best_feature, 'value': best_split_val, 'Gini': best_gini,
            'groups': best_groups, 'group_labels': best_group_labels}

    return node


def terminate_node(group_labels):
    group_size = len(group_labels)
    classVals = range(int(np.max(group_labels) + 1))
    probs = [0, 0]  # holds the probs of each class
    counts = [0, 0]  # holds how many of each class is in the group
    for classVal in classVals:
        probs[classVal] = (np.sum(group_labels == classVal)/group_size)
        counts[classVal] = (np.sum(group_labels == classVal))

    return {'probs': probs, 'counts': counts}


def split_tree(node, max_depth, min_size, depth):
    # RECURSION!!!

    left_group, right_group = node['groups']
    left_group_labels, right_group_labels = node['group_labels']
    del(node['groups'])  # delete the old groups from the node
    # check if either of the groups are empty
    if left_group.size == 0 or right_group.size == 0:
        # use the non-empty group to terminate
        if left_group.size > 0:
            node['left'] = node['right'] = terminate_node(left_group_labels)
            return

        if right_group.size > 0:
            node['left'] = node['right'] = terminate_node(right_group_labels)
            return

    # check if max depth exceeded
    if depth >= max_depth:
        node['left'], node['right'] = terminate_node(left_group_labels), terminate_node(right_group_labels)
        return

    # split left child
    if len(left_group) <= min_size:
        node['left'] = terminate_node(left_group_labels)
    else:
        node['left'] = get_best_split(left_group, left_group_labels)
        split_tree(node['left'], max_depth, min_size, depth+1)

    # split right child
    if len(right_group) <= min_size:
        node['right'] = terminate_node(right_group_labels)
    else:
        node['right'] = get_best_split(right_group, right_group_labels)
        split_tree(node['right'], max_depth, min_size, depth+1)


def train(train_labels, train_features, max_depth, min_size):
    root = get_best_split(train_features, train_labels)
    split_tree(root, max_depth, min_size, 1)
    return root


def print_tree(node, depth=0):
    if 'feature' in node:
        print('%s[X%d < %.3f]' % (depth*' ', (node['feature']+1), node['value']))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))


def predict(node, row):
    if row[node['feature']] < node['value']:
        if 'feature' in node['left']:
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if 'feature' in node['right']:
            return predict(node['right'], row)
        else:
            return node['right']


def run(tree, features):
    probs = []
    for i in range(len(features)):
        prediction = predict(tree, features[i, :])
        probs.append(prediction['probs'])

    return np.asarray(probs)


if __name__ == "__main__":
    # test the gini_index function
    test_groups = [[1, 1], [1, 1]]
    test_group_labels = [[0, 1], [0, 1]]

    print(gini_index(test_groups, test_group_labels))

    data = [[2.771244718, 1.784783929, 0],
            [1.728571309, 1.169761413, 0],
            [3.678319846, 2.81281357, 0],
            [3.961043357, 2.61995032, 0],
            [2.999208922, 2.209014212, 0],
            [7.497545867, 3.162953546, 1],
            [9.00220326, 3.339047188, 1],
            [7.444542326, 0.476683375, 1],
            [10.12493903, 3.234550982, 1],
            [6.642287351, 3.319983761, 1]]
    data = np.asarray(data)
    test_features = data[:, :-1]
    test_labels = data[:, -1]

    # test get_best_split function
    # node = get_best_split(test_features, test_labels)
    # print(node)
    # print(terminate_node(test_labels))

    tree = train(test_features,test_labels, 2, 1)
    print(run(tree, test_features))
    print_tree(tree)

