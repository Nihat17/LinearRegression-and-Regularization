#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:48:53 2018

@author: Nihat Allahverdiyev
"""

import mglearn.plots
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples = 60)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state = 42)

lr = LinearRegression().fit(X_train, y_train)

print("Coefficient: {}".format(lr.coef_))  # weight
print("Coefficient: {}".format(lr.intercept_))  # b

print("Accuracy of training: {:.2f}".format(lr.score(X_train, y_train)))
print("Accuracy of testing: {:.2f}".format(lr.score(X_test, y_test)))

    #Result means underFitting
