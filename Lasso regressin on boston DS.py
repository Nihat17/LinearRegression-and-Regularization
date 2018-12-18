#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:24:04 2018

@author: Nihat Allahverdiyev
"""

from sklearn.linear_model import Lasso
import mglearn.datasets as md
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

X, y = md.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state = 0)
ls = Lasso().fit(X_train, y_train)

print("Training score: {:.2f}".format(ls.score(X_train, y_train)))
print("Test score: {:.2f}".format(ls.score(X_test, y_test)))
print("Number of features: {}".format(np.sum(ls.coef_ != 0)))

# we increase the default setting of "max_iter"
# otherwise the model would warn us that we should increase "max_iter"
lasso001 = Lasso(alpha = 0.01, max_iter = 100000).fit(X_train, y_train)

print("Training score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features: {}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha = 0.0001, max_iter = 100000).fit(X_train, y_train)

print("Training score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features: {}".format(np.sum(lasso00001.coef_ != 0)))

plt.plot(ls.coef_, 's', label = "Lasso alpha = 1")
plt.plot(lasso001.coef_, '^', label = "alpha = 0.01")
plt.plot(lasso00001.coef_, 'v', label = "alpha = 0.0001")
plt.legend(ncol = 2, loc = (0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")

# ElasticNet combines the penalties of Ridge and Lasso 
# By using it we are getting slightly better result
regr = ElasticNet(alpha = 0.001, max_iter = 100000)
regr.fit(X_train, y_train)

print("Training score: {:.2f}".format(regr.score(X_train, y_train)))
print("Test score: {:.2f}".format(regr.score(X_test, y_test)))
print("Number of features: {}".format(np.sum(regr.coef_ != 0)))










