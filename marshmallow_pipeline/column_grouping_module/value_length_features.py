import pandas as pd
from numpy import median, std
from sklearn.base import BaseEstimator, TransformerMixin


class ValueLengthStats(BaseEstimator, TransformerMixin):
    """
    Computes the median and standard deviation of the number of characters in each cell of a DataFrame
    """

    def __init__(self, weight):
        self.weight = weight
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Compute the number of characters in each cell of the DataFrame
        char_counts = [[len(str(x).strip()) for x in col] for col in X]

        char_counts_median = [
            median(char_counts_col) * self.weight for char_counts_col in char_counts
        ]
        char_counts_std = [
            std(char_counts_col) * self.weight for char_counts_col in char_counts
        ]
        df = pd.DataFrame({"median": char_counts_median, "std": char_counts_std})
        return df
