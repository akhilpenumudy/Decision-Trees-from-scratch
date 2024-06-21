import numpy as np
from collections import Counter


class Node:
    """
    A node in the decision tree.

    Attributes:
        feature (int): The index of the feature to split on.
        threshold (float): The threshold value to split the feature.
        left (Node): The left child node.
        right (Node): The right child node.
        value (int): The class label if the node is a leaf.
    """

    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if the node is a leaf node."""
        return self.value is not None


class DecisionTree:
    """
    A decision tree classifier.

    Attributes:
        min_samples_split (int): The minimum number of samples required to split a node.
        max_depth (int): The maximum depth of the tree.
        n_features (int): The number of features to consider when looking for the best split.
        root (Node): The root node of the tree.
    """

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Build the decision tree.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target vector.
        """
        self.n_features = self.n_features or X.shape[1]
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grow the tree.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target vector.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the (sub)tree.
        """
        n_samples, n_labels = len(y), len(np.unique(y))
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            return Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """
        Find the best split for the given data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target vector.
            feat_idxs (array-like): Indices of features to consider for the best split.

        Returns:
            tuple: The best feature index and the best threshold for the split.
        """
        best_gain, split_idx, split_thresh = -1, None, None
        for feat_idx in feat_idxs:
            X_col, thresholds = X[:, feat_idx], np.unique(X[:, feat_idx])
            for thr in thresholds:
                gain = self._information_gain(y, X_col, thr)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thr
        return split_idx, split_thresh

    def _information_gain(self, y, X_col, threshold):
        """
        Calculate the information gain of a potential split.

        Parameters:
            y (array-like): Target vector.
            X_col (array-like): Feature column.
            threshold (float): The threshold to split the feature column.

        Returns:
            float: The information gain of the split.
        """
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_col, threshold)
        if not len(left_idxs) or not len(right_idxs):
            return 0
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        return parent_entropy - (n_l / n) * e_l - (n_r / n) * e_r

    def _split(self, X_col, split_thresh):
        """
        Split the data based on the given threshold.

        Parameters:
            X_col (array-like): Feature column.
            split_thresh (float): The threshold to split the feature column.

        Returns:
            tuple: Indices of the left and right splits.
        """
        left_idxs = np.argwhere(X_col <= split_thresh).flatten()
        right_idxs = np.argwhere(X_col > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculate the entropy of the target vector.

        Parameters:
            y (array-like): Target vector.

        Returns:
            float: The entropy of the target vector.
        """
        hist, ps = np.bincount(y), np.bincount(y) / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Find the most common label in the target vector.

        Parameters:
            y (array-like): Target vector.

        Returns:
            int: The most common label.
        """
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
            X (array-like): Feature matrix.

        Returns:
            array: Predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction.

        Parameters:
            x (array-like): Single data point.
            node (Node): Current node.

        Returns:
            int: Predicted class label.
        """
        if node.is_leaf_node():
            return node.value
        return self._traverse_tree(
            x, node.left if x[node.feature] <= node.threshold else node.right
        )
