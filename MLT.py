from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import linear_model


train = pd.read_csv("train_shuffeld.csv")
test = pd.read_csv("test_shuffeld.csv")
train_X = train[["activity","screen","valence","arousal","office","game","call","sms","social","entertainment","communication","weather"]]
train_Y = train[["mood"]]
test_X = test[["activity","screen","valence","arousal","office","game","call","sms","social","entertainment","communication","weather"]]
test_Y = test[["mood"]].to_numpy()
criterions = ["absolute_error"]
splitters = ["random"]
max_depths = [8]
min_samples_splits = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
min_samples_leafs = [19]
min_weight_fraction_leafs = [0.1]
max_features = ["log2"]
max_leaf_nodes = [None,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
min_impurity_decreases = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
best_model = 0
best_errors = [np.inf]
best_parameters = []
for criterion in criterions:
    for splitter in splitters:
        for max_depth in max_depths:
            for min_samples_leaf in min_samples_leafs:
                for min_weight_fraction_leaf in min_weight_fraction_leafs:
                    for max_feature in max_features:
                        for max_leaf_node in max_leaf_nodes:
                            for min_impurity_decrease in min_impurity_decreases:
                                model = tree.DecisionTreeRegressor(criterion=criterion,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                                                   min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_feature,
                                                                   max_leaf_nodes=max_leaf_node, min_impurity_decrease=min_impurity_decrease)
                                model.fit(train_X.to_numpy(),train_Y.to_numpy())
                                predictions = model.predict(test_X.to_numpy())
                                correct = 0
                                error = []
                                for i in range(len(predictions)):
                                    if test_Y[i] == predictions[i]:
                                        correct = correct + 1
                                    error.append(float(abs(test_Y[i]-predictions[i])))
                                if np.mean(error) < np.mean(best_errors):
                                    print(np.mean(error))
                                    best_errors = error
                                    best_model = model
                                    best_parameters = [criterion,splitter,max_depth,min_samples_leaf,min_weight_fraction_leaf,max_feature,max_leaf_node,min_impurity_decrease,correct,np.mean(error)]
                                    print(best_parameters)
with open("tree_errors.txt","w") as error_file:
    for value in best_errors:
        error_file.write(str(value))
        error_file.write("\n")