import logging

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def classify(X_train, y_train, X_test):
    predicted = []
    gbc = None
    logging.debug("Classification")
    if sum(y_train) == 0:
        predicted = [0] * len(X_test)
    elif sum(y_train) == len(y_train):
        predicted = [1] * len(X_test)
    else:
        gbc = GradientBoostingClassifier(n_estimators=100)
        gbc.fit(X_train, y_train)
        if len(X_test) > 0:
            predicted = gbc.predict(X_test)
    return gbc, predicted

def classify_with_cs(X_train, y_train, X_test):
    predicted = []
    gbc = None
    logging.debug("Classification")
    if sum(y_train) == 0:
        predicted = [0] * len(X_test)
    elif sum(y_train) == len(y_train):
        predicted = [1] * len(X_test)
    else:
        svc = SVC(kernel='rbf', probability=True, class_weight='balanced')
        svc.fit(X_train, y_train)
        if len(X_test) > 0:
            predicted_classes = svc.predict(X_test)
            # Predicting the probabilities
            predicted_probabilities = svc.predict_proba(X_test)
            # Extracting the probabilities of the predicted classes
            probabilities_of_predicted_classes = np.array([prob[class_idx] for prob, class_idx in zip(predicted_probabilities, predicted_classes)])
            print(probabilities_of_predicted_classes)

    return svc, predicted_classes
