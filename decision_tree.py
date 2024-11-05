import sys
import csv
from logistic_regression import logistic_regression  
import pandas as pd
import numpy as np


class ObliqueDecisionTree:
    def __init__(self, max_depth, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.node_info = []  

    class Node:
        def __init__(self, depth=0, node_id=None):
            self.depth = depth
            self.node_id = node_id  
            self.is_leaf = False
            self.split_value = None
            self.feature_combination = None
            self.left = None
            self.right = None
            self.prediction = None
            self.weights = None 

    def _custom_logistic_regression_split(self, X, y):
        feature_combination = logistic_regression(X, y)  
        linear_combination = X.dot(feature_combination)
        return linear_combination, feature_combination

    def _gini_impurity(self, y):
        m = len(y)
        if m == 0:
            return 0
        p1 = np.sum(y) / m
        p0 = 1 - p1
        return 1 - p1**2 - p0**2

    def _best_threshold(self, linear_combination, y):
        sort_idx = np.argsort(linear_combination)
        sorted_combination = linear_combination[sort_idx]
        sorted_y = y[sort_idx]
        
        best_gini = float('inf')
        best_threshold = None
        n = len(y)

        for i in range(1, n):
            left_y, right_y = sorted_y[:i], sorted_y[i:] 
            gini_left = self._gini_impurity(left_y)
            gini_right = self._gini_impurity(right_y)
            weighted_gini = (len(left_y) * gini_left + len(right_y) * gini_right) / n
            
            if weighted_gini < best_gini and sorted_combination[i-1]!=sorted_combination[i]:
                best_gini = weighted_gini
                best_threshold = (sorted_combination[i - 1] + sorted_combination[i]) / 2
                
        return best_threshold, best_gini

    def _build_tree(self, X, y, depth=0, node_id=1):
        node = self.Node(depth, node_id)
        
        if (self.max_depth and depth >= self.max_depth) or len(y) < self.min_samples_split or np.all(y == y[0]):
            node.is_leaf = True
            node.prediction = np.round(np.mean(y)).astype(int)
            return node

        linear_combination, feature_combination = self._custom_logistic_regression_split(X, y)
        best_threshold, best_gini = self._best_threshold(linear_combination, y)

        if best_threshold == None:
            node.is_leaf = True
            node.prediction = np.round(np.mean(y)).astype(int)
            return node

        if best_gini == 1.0:
            node.is_leaf = True
            node.prediction = np.round(np.mean(y)).astype(int)
            return node

        left_mask = linear_combination <= best_threshold
        right_mask = linear_combination > best_threshold
        
        node.split_value = best_threshold
        node.feature_combination = feature_combination
        node.weights = feature_combination 

        self.node_info.append([node.node_id, node.weights, node.split_value])

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1, 2 * node_id)  
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1, 2 * node_id + 1)  
        
        return node

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_sample(self, x, node):
        if node.is_leaf:
            return node.prediction
        linear_combination = np.dot(x, node.feature_combination)
        if linear_combination <= node.split_value:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def save_weights_to_csv(self, filename):
        formatted_node_info = []
        
        def collect_node_info(node):
            if node.is_leaf:
                return  
            weight_list = node.weights.flatten().tolist() if node.weights is not None else []
            formatted_node_info.append([node.node_id] + weight_list + [node.split_value])
            
            if node.left:
                collect_node_info(node.left)
            if node.right:
                collect_node_info(node.right)

        collect_node_info(self.tree)

        df = pd.DataFrame(formatted_node_info)
        df.to_csv(filename, index=False, header=False)

    def prune_tree(self, X_val, y_val):
        def _prune(node, X_subset, y_subset):
            if node.is_leaf:
                return 
            if(len(y_subset)==0):
                node.is_leaf = True
                node.prediction = 0
                node.left = None
                node.right = None
                return
            linear_combination = np.dot(X_subset, node.feature_combination)
            left_mask = linear_combination <= node.split_value
            right_mask = linear_combination > node.split_value
            
            X_left = X_subset[left_mask]
            y_left = y_subset[left_mask]
            X_right = X_subset[right_mask]
            y_right = y_subset[right_mask]

            if node.left is not None:
                _prune(node.left, X_left, y_left)
            if node.right is not None:
                _prune(node.right, X_right, y_right)

            original_predictions = self._predict_node(node, X_subset)
            original_accuracy = np.mean(original_predictions == y_subset)
            majority_class = np.round(np.mean(y_subset)).astype(int)

            new_predictions = np.full(y_subset.shape, majority_class)
            new_accuracy = np.mean(new_predictions == y_subset)

            if new_accuracy >= original_accuracy:
                node.is_leaf = True
                node.prediction = majority_class
                node.left = None
                node.right = None

        _prune(self.tree, X_val, y_val)

    def _predict_node(self, node, X):
        if node.is_leaf:
            return np.full(X.shape[0], node.prediction)
        else:
            linear_combination = np.dot(X, node.feature_combination)
            left_mask = linear_combination <= node.split_value
            right_mask = linear_combination > node.split_value
            
            left_predictions = self._predict_node(node.left, X[left_mask])
            right_predictions = self._predict_node(node.right, X[right_mask])
            
            predictions = np.empty(X.shape[0])
            predictions[left_mask] = left_predictions
            predictions[right_mask] = right_predictions
            
            return predictions


def train_unpruned(train_file, max_depth, weight_file):
    data = pd.read_csv(train_file)
    X = data.drop('target', axis=1).values
    Y = data['target'].values
    tree = ObliqueDecisionTree(max_depth=max_depth)
    tree.fit(X, Y)
    tree.save_weights_to_csv(weight_file)


def train_pruned(train_file, val_file, max_depth, weight_file):
    data = pd.read_csv(train_file)
    X = data.drop('target', axis=1).values
    Y = data['target'].values

    val_data = pd.read_csv(val_file)
    X_val = val_data.drop('target', axis=1).values
    Y_val = val_data['target'].values

    tree = ObliqueDecisionTree(max_depth=max_depth)
    tree.fit(X, Y)
    tree.prune_tree(X_val, Y_val)
    tree.save_weights_to_csv(weight_file)


def test_pruned(train_file, val_file, test_file, max_depth, prediction_file):
    data = pd.read_csv(train_file)
    X = data.drop('target', axis=1).values
    Y = data['target'].values

    val_data = pd.read_csv(val_file)
    X_val = val_data.drop('target', axis=1).values
    Y_val = val_data['target'].values

    tree = ObliqueDecisionTree(max_depth=max_depth)
    tree.fit(X, Y)
    
    tree.prune_tree(X_val, Y_val)
    
    test_data = pd.read_csv(test_file)
    if 'target' in test_data.columns:
        X_test = test_data.drop('target', axis=1).values
    else:
        X_test = test_data

    Y_test= tree.predict(X_test)

    with open(prediction_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for prediction in Y_test:
            writer.writerow([prediction]) 



def main():
    if len(sys.argv) < 5:
        print("Invalid arguments. Check command format.")
        sys.exit(1)

    command_type = sys.argv[1]
    mode = sys.argv[2]
    if command_type == "train":
        if mode == "unpruned":
            train_file = sys.argv[3]
            max_depth = int(sys.argv[4])
            weight_file = sys.argv[5]
            train_unpruned(train_file, max_depth, weight_file)
        elif mode == "pruned":
            train_file = sys.argv[3]
            val_file = sys.argv[4]
            max_depth = int(sys.argv[5])
            weight_file = sys.argv[6]
            train_pruned(train_file, val_file, max_depth, weight_file)
    elif command_type == "test":
        train_file = sys.argv[2]
        val_file = sys.argv[3]
        test_file = sys.argv[4]
        max_depth = int(sys.argv[5])
        prediction_file = sys.argv[6]
        test_pruned(train_file, val_file, test_file, max_depth, prediction_file)
    else:
        print("Invalid command type.")
        sys.exit(1)


if __name__ == "__main__":
    main()
