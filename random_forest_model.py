#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 02:26:33 2021

@author: marekdziergas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import random

################
# training set #
################

# Processed taining data set
training_set = pd.read_csv("train_set.csv",sep = ',')

# Labels are the values we want to predict
training_target = np.array(training_set['innum'])

# Remove the target variable from predictors
# axis 1 refers to the columns
# Contains ID column for later reference
training_predictors = training_set.drop('innum', axis = 1)
training_predictors = training_predictors.drop('data_source', axis = 1)
# Remove ID column
training_predictors_wo_id = training_predictors.drop('ID', axis = 1)

# Convert to numpy array
training_predictors = np.array(training_predictors)
predictors_wo_id = np.array(training_predictors_wo_id)


##################
# validation set #
##################
# Processed validation/test set
validation_set = pd.read_csv("test_set.csv",sep = ',')

# Remove the target variable from predictors
# axis 1 refers to the columns
# Contains ID column for later reference
validation_predictors = validation_set.drop('innum', axis=1)
validation_predictors = validation_predictors.drop('data_source', axis=1)

# Remove ID column
validation_predictors_wo_id = validation_predictors.drop('ID', axis=1)

# Convert to numpy array
validation_predictors = np.array(validation_predictors)
validation_predictors_wo_id = np.array(validation_predictors_wo_id)


######################
# Split the data set #
######################

# split the data, leave ID column
X_train, X_test, y_train, y_test = train_test_split(training_predictors, training_target, test_size=.05, random_state=875)

# remove ID column
X_train_wo_id = np.delete(X_train, 0, 1)
X_test_wo_id =np.delete(X_test, 0, 1)


def randomForest(X, y):
    clf = RandomForestRegressor(n_estimators=100)
    clf = clf.fit(X, y)
    return clf
def gradientBoosting(X, y):
    clf = GradientBoostingRegressor(subsample=.8, n_estimators = 100)
    clf = clf.fit(X, y)
    return clf
def neuralNet(X, y):
    clf = MLPRegressor(batch_size=64, max_iter=200, hidden_layer_sizes=(250, 120, 50), learning_rate_init=0.05)
    clf = clf.fit(X, y)
    return clf


random_forest_model = randomForest(X_train_wo_id, y_train)
rf_y_predictions = random_forest_model.predict(X_test_wo_id)


rf_mae = mean_absolute_error(y_test, rf_y_predictions)

print(rf_mae)

gb = gradientBoosting(X_train_wo_id, y_train)
gb_y_predictions = gb.predict(X_test_wo_id)

gb_mae = mean_absolute_error(y_test, gb_y_predictions)
print(gb_mae)

nn = neuralNet(X_train_wo_id, y_train)
nn_y_predictions = nn.predict(X_test_wo_id)

nn_mae = mean_absolute_error(y_test, nn_y_predictions)
print(nn_mae)


################
# GridSearchCV #
################

param_list = {"hidden_layer_sizes": [(250, 120, 50), (250,120,50,25)],"learning_rate_init":[.005, .01, .05], "batch_size":[16, 32, 64, 'auto'], "max_iter":[200]}

kf = KFold(n_splits=3)

neural = MLPRegressor()
clf = GridSearchCV(estimator=neural, param_grid=param_list, scoring='neg_mean_absolute_error', cv = kf)
fitted = clf.fit(X_train_wo_id, y_train)
print(clf.best_estimator_)
print(clf.best_score_)


'''
def hypertune(X_train_wo_id, y_train):
    lowest_mae = 100000
    min_split = [2, 4, 6]
    max_depth = [25, 30, 35, 40]
    max_samples = [.7,.8,.9]
    max_features = [15, 68, 78]
    randomlist = random.sample(range(10, 1000), 1)
    for i in randomlist:
        for md in max_depth:
            for ms in max_samples:
                for mp in min_split:
                    for mf in max_features:
                        random_forest_model = gradientBoosting(X_train_wo_id, y_train, i, md, ms, mp,mf)
                        rf_y_predictions = random_forest_model.predict(X_test_wo_id)
                        rf_mae = mean_absolute_error(y_test, rf_y_predictions)
                        if rf_mae < lowest_mae:
                            lowest_mae = rf_mae
                            lowest_params = []
                            lowest_params.append([f"MAE: {rf_mae}", f"Min Split:{mp}", f"Max Depth: {md}", f"Max/Sub Samples: {ms}", f"Max Features: {mf}",f"seed: {i}"])
                        print(f"MAE: {rf_mae} | Min Split:{mp} | Max Depth: {md} | Max/Sub Samples: {ms} | Max Features: {mf} | seed: {i}")
                        print(lowest_mae, lowest_params)
    print(lowest_mae)
    print(lowest_params)
hypertune(X_train_wo_id, y_train)

'''
# validation set output
valid_predictions = []
for i in range(len(validation_predictors_wo_id)):
    valid_pred = random_forest_model.predict([validation_predictors_wo_id[i]])
    mall_id = validation_predictors[i][0]
    valid_predictions.append([mall_id, valid_pred[0]])
    
np.savetxt("neural_net_val.csv", 
           valid_predictions,
           delimiter =", ", 
           fmt ='% s')


