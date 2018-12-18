#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 00:27:45 2018

@author: Nihat Allahverdiyev
"""
from sklearn.model_selection import train_test_split
import mglearn.datasets as md
from sklearn.linear_model import LinearRegression

# load the data
X, y = md.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state = 0)

lr = LinearRegression().fit(X_train, y_train)

print("Accuracy of predicting training: {:.2f}".format(lr.score(
        X_train, y_train)))


print("Accuracy of predicting test: {:.2f}".format(lr.score(
        X_test, y_test)))
 
    # Result means Overfitting