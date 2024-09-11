import numpy as np 
import graphviz

import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature_index = None, split_value= None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.split_value = split_value
        self.left = left
        self.right = right
        self.value = value

class ConformalTree: 
    def __init__(self, model, alpha, min_samples_leaf, max_depth, x_cal, y_cal): 
        self.model = model 
        self.alpha = alpha
        self.tree = None
        self.depth = 0
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def _compute_conformal_scores(self, x_cal, y_cal):
        conformal_scores = self.model.predict(x_cal) - y_cal
        return conformal_scores
    
    def fit(self, X, y):
        conformal_scores =  self._compute_conformal_scores(X, y)
        self.tree = self._build_tree(X, y, conformal_scores)

    def _build_tree(self, X, y, conformal_scores): 
        self.depth += 1
        if len(y) == 1:
            Warning("A qunatile was estimated for n_sample=1, increase the min_samples_leaf or reduce the max_depth")
            return(Node(value = y[0])) 
        elif (len(y) <= self.min_samples_leaf) or (self.depth >= self.max_depth):
            return Node(value=np.quantile(conformal_scores, 1-self.alpha)) 
        best_feature, best_split = self._find_best_split(X, y)

        Left_mask = X[:,best_feature] <= best_split
        Right_mask = ~Left_mask

        if np.sum(Left_mask) < self.min_samples_leaf or np.sum(Right_mask) < self.min_samples_leaf:
            return Node(value=np.quantile(conformal_scores, 1-self.alpha))

        left_node = self._build_tree(X[Left_mask,:], y[Left_mask], conformal_scores[Left_mask])
        right_node = self._build_tree(X[Right_mask,:], y[Right_mask], conformal_scores[Right_mask])

        return Node(feature_index=best_feature, split_value=best_split, left=left_node, right=right_node)

    def _find_best_split(self, X, y):
        best_value = 1
        best_feature = None
        best_var = float('inf')


        n_features = X.shape[1]
        for feature_idx in range(n_features):
            unique_values = np.unique(X[:,feature_idx])
            for value in unique_values:
                left_mask = X[:,feature_idx] <= value
                right_mask = ~left_mask

                if np.any(left_mask) and np.any(right_mask):
                    left_y = y[left_mask]
                    right_y = y[right_mask]

                    weighted_var = (len(left_y) * np.var(left_y) + len(right_y) * np.var(right_y))/len(y)

                if weighted_var < best_var:
                    best_feature = feature_idx
                    best_value = value
                    best_var = weighted_var
            
        return best_feature, best_value
    
    def predict(self, X): 
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, node): 
        if node.value is not None: 
            prediction = self.model.predict(sample.reshape(1,-1))
            lower = prediction - node.value
            upper = prediction + node.value
            return lower, upper    
        if sample[node.feature_index] <= node.split_value:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
    
    def plot_tree(self):
        dot = graphviz.Digraph()
        self._plot_node(self.tree, dot)
        dot.render('regression_tree', format='png', cleanup=True)
        dot.view('regression_tree')

    def _plot_node(self, node, dot, node_id=0):
        if node is None:
            return

        node_label = f"Value: {node.value:.2f}" if node.value is not None else f"Feature {node.feature_index} <= {node.split_value:.2f}"
        dot.node(str(node_id), label=node_label)

        if node.left is not None:
            dot.edge(str(node_id), str(node_id * 2 + 1), label='True')
            self._plot_node(node.left, dot, node_id * 2 + 1)

        if node.right is not None:
            dot.edge(str(node_id), str(node_id * 2 + 2), label='False')
            self._plot_node(node.right, dot, node_id * 2 + 2)

    def get_split_values(self, sort = True):
        return np.sort(self._get_split_values(self.tree)) if sort else np.array(self._get_split_values(self.tree))
    
    def _get_split_values(self, node):
        if node.value is not None:
            return []
        return [node.split_value] + self._get_split_values(node.left) + self._get_split_values(node.right)

def ploting_utility(x_test, y_test, model, tree, title):
    intervals = tree.get_split_values()
    intervals = np.insert(intervals, 0, x_test.min())
    intervals = np.append(intervals, x_test.max())
    
    predictions = np.squeeze(tree.predict(x_test))
    lower = predictions[:,0]
    upper = predictions[:,1]

    sort_test = np.argsort(x_test, axis=0).flatten()
    plt.figure(figsize=(10,5))
    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test[sort_test], model.predict(x_test[sort_test]), color='blue', label='Model Prediction')
    plt.fill_between(x_test[sort_test].ravel(), lower[sort_test], upper[sort_test], color='red', alpha=0.5, label='Conformal Prediction')
    for i in range(len(intervals) - 1):
        color = 'gray' if i % 2 == 0 else 'white'
        plt.axvspan(intervals[i], intervals[i+1], color=color, alpha=0.2)
    plt.title(title)


# Example usage
if __name__ == "__main__":
    from sklearn.linear_model import HuberRegressor
    from generate import generate_cqr_data

    x_train, y_train, x_cal, y_cal, x_test, y_test = generate_cqr_data(0)
    model = HuberRegressor()
    model.fit(x_train, y_train)
    tree = ConformalTree(alpha=0.95,
                         model=model,
                         min_samples_leaf=10, 
                         max_depth=1000,
                         x_cal=x_cal,
                         y_cal=y_cal)
    tree.fit(x_cal, y_cal)
    tree.plot_tree()
    ploting_utility(x_test, y_test, model, tree, 'Model Prediction')
    plt.ylim(-5, 10)
    plt.show()
    print("Done!")
