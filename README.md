# Iris

## Data Visualization


![pair plot](https://github.com/QQQQ00243/Iris/blob/main/images/pairplot.png)

## Stratified Shuffle Split

## Various Classification Methods

### SGD Classification(accuracy=76.7%)

![SGD Classifier Learning Curves](https://github.com/QQQQ00243/Iris/blob/main/images/SGD_learning_curves.png)

### Softmax Classification(accuracy=96.7%)

![softmax_learning_curves](.\images\softmax_learning_curves.png)

### DecisionTree

#### Before Fine Tuning(accuracy=93.3%)

![learning_curves_decision_tree_before_fine_tuning](https://github.com/QQQQ00243/Iris/blob/main/images/decision_tree_learning_curves_before.png)

Strongly overfitted!

What the tree looks like

![tree_before_tuning](https://github.com/QQQQ00243/Iris/blob/main/images/tree_before_tuning.png) 

#### After Fine Tuning(accuracy=97.3%)

parameters: {'criterion': 'gini', 'max_depth': 3, 'max_features': 4, 'max_leaf_nodes': 5}

![learning_curves_best_tree_classifier](.\images\decision_tree_learning_curves_after.png)

![best_tree_plot](.\images\tree_after_tuning.png)

## Error Analysis

We notice that there is no classification error between setosa and the other two classes. By looking at the scatter plot we can tell immediately that the cluster of setosa do not intersect with the other two.  Instances at the intersection of the clusters of versicolor and verginica.

![Confusion matrix](.\images\conf_mx.png)

![Error](.\images\error_pairplot.png)



