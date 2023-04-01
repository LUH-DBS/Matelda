import pandas as pd
from numpy import median, std
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CharDistribution(BaseEstimator, TransformerMixin):
    """
    Computes the character distribution of each column
    """
    def __init__(self, char_set, weight):
        self.char_set = char_set
        self.weight = weight
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        char_distributions = []
        for col in X:
            # Define a default dict to keep track of the counts for each type
            char_observations = {char: 0 for char in self.char_set}

            # Loop through the list and count the number of instances per type
            for value in col:
                for char in set(str(value)):
                    char_observations[char] += 1
            char_observations = {k: (v/len(col)) * self.weight for k, v in char_observations.items()}
            char_distributions.append(char_observations)
        char_distributions = pd.DataFrame(char_distributions)
        return char_distributions