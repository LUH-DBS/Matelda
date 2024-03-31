import logging
from sklearn.ensemble import GradientBoostingClassifier


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
