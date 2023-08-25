import logging

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def classify(X_train, y_train, X_test):
    logging.debug("Classification")
    if sum(y_train) == 0:
        predicted = [0] * len(X_test)
    elif sum(y_train) == len(y_train):
        predicted = [1] * len(X_test)
    else:
        gbc = GradientBoostingClassifier(n_estimators=100)
        gbc.fit(X_train, y_train)
        predicted = gbc.predict(X_test)
    return predicted
