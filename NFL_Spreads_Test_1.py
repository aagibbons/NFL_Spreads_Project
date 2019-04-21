# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:57:02 2019

@author: Owner
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time

#time counter
start_time = time.time()

# import the data
df = pd.read_csv("Spreads_Test_1.csv")
data = np.array(df)

# split into training and test portions
train = data[:1439,:]
test = data[1439:,:]

# separate the labels from the the features
train_features = train[:,1:]
test_features = test[:,1:]
train_labels = train[:,0]
test_labels = test[:,0]



# Instantiate model with 1000 decision trees
for min_samples_split in range(100,121,5):

    if min_samples_split == 0:
        min_samples_split = 2
        
    for max_depth in range(2,11,2):
    
        for b in range(1):
            print()
            rf = RandomForestRegressor(n_estimators=1000, max_depth=max_depth, min_samples_split=min_samples_split,
                                       random_state=b)
        
            # Train the model on training data
            rf.fit(train_features, train_labels);
            
            # Use the forest's predict method on the test data
            predictions = rf.predict(test_features)
            
            # build matrix to keep track of predictions
            if b == 0:
                predictions_matrix = np.array(predictions)
            else:
                predictions_matrix = np.column_stack((predictions_matrix, predictions))
        
        # calculate errors and root mean square error
        if len(np.shape(predictions_matrix)) == 1:
            errors = (predictions_matrix - test_labels) ** 2
        else:
            errors = (np.average(predictions_matrix, axis=1) - test_labels) ** 2
        rmse = (np.average(errors)) ** 0.5
        root_mean_square_error = round(rmse, 4)
        
        print("MD", max_depth, "MSS", min_samples_split, "RMSE", root_mean_square_error)



print("My program took", int(time.time() - start_time), "to run")




