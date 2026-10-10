import numpy as np
import pandas as pd

# Building ML models
def knn(x_train, y_train, x, k):
    features = np.array(x_train)
    targets = np.array(y_train)

    distance = np.linalg.norm(features - x, axis=1)

    numbers = pd.DataFrame({"points": distance, "class": targets})

    numbers = numbers.sort_values("points")
    numbers = numbers[:k]

    return numbers["class"].value_counts().idxmax()

def linear_regression(x_train, y_train, alpha, iterations):
    y_train = np.array(y_train)
    w = np.zeros(x_train.shape[1])
    b = 0
    for _ in range(iterations):
        y = x_train @ w + b
        gradient_w = (2 / len(y_train)) * (x_train.T @ (y - y_train))
        gradient_b = (2 / len(y_train)) * np.sum(y - y_train)
        w = w - gradient_w * alpha
        b = b - gradient_b * alpha
    return w, b

def classification(x_train, y_train, x, alpha):
    w = np.zeros(x_train.shape[1])
    b = 0
    z = x_train @ w + b
    p = 1 / (1 + np.e**(-z))
    for _ in range(1000):
        gradient_w = (x_train.T @ (p - y_train)) / len(y_train)
        gradient_b = (p - y_train).mean()
        b = b - gradient_b * alpha
        w = w - gradient_w * alpha
        z = x_train @ w + b
        p = 1 / (1 + np.e**(-z))

    z = x @ w + b
    p = 1 / (1 + np.e**(-z))
    return p, w, b

def decision_tree(x_train=None, y_train=None, tree=None, x=None):
    if tree is not None:
        if str(type(tree)) != "<class 'dict'>":
            return tree
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return decision_tree(tree=tree["left"], x=x)
        else:
            return decision_tree(tree=tree["right"], x=x)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    if len(y_train) <= 1:
        return np.mean(y_train)
    if len(np.unique(y_train)) == 1:
        return np.mean(y_train)
    best_sse = np.inf
    best_feature = None
    best_threshold = None
    for feature in range(x_train.shape[1]):
        values = np.unique(x_train[:, feature])

        if len(values) <= 1:
            continue
        thresholds = (values[:-1] + values[1:]) / 2
        for threshold in thresholds:
            left_mask = x_train[:, feature] <= threshold
            right_mask = x_train[:, feature] > threshold
            if not np.any(left_mask) or not np.any(right_mask):
                continue
            left_y = y_train[left_mask]
            right_y = y_train[right_mask]
            avg_left_y = np.mean(left_y)
            avg_right_y = np.mean(right_y)
            error_left = np.sum((left_y - avg_left_y) ** 2)
            error_right = np.sum((right_y - avg_right_y) ** 2)
            sse = error_left + error_right
            if sse < best_sse:
                best_sse = sse
                best_feature = feature
                best_threshold = threshold
    if best_feature is None:
        return np.mean(y_train)
    left_mask = x_train[:, best_feature] <= best_threshold
    right_mask = x_train[:, best_feature] > best_threshold
    left_x = x_train[left_mask]
    right_x = x_train[right_mask]
    left_y = y_train[left_mask]
    right_y = y_train[right_mask]
    left_tree = decision_tree(x_train=left_x, y_train=left_y)
    right_tree = decision_tree(x_train=right_x, y_train=right_y)

    return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}

def random_forest(x_train=None, y_train=None, n_trees=None, trees=None, x=None):
    if trees is not None:
        predictions = []

        for tree in trees:
            prediction = decision_tree(tree=tree, x=x)
            predictions.append(prediction)

        return np.mean(predictions)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    trees = []
    for _ in range(n_trees):
        indices = np.random.choice(len(x_train), size=len(x_train), replace=True)
        x_tree = x_train[indices]
        y_tree = y_train[indices]
        tree = decision_tree(x_train=x_tree, y_train=y_tree)
        trees.append(tree)

    return trees
