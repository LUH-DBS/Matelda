import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from dateutil.parser import parse
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class SingleDataTypeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # The fit method does not need to do anything
        return self

    def transform(self, X):
        features_df = None
        # Loop through the columns and compute the data type features for each one
        all_types = ["int", "float", "complex", "bool", "str", "datetime"]
        all_cols_types = []
        for col in X:
            # Define a default dict to keep track of the counts for each type
            type_counts = {data_type: 0 for data_type in all_types}
            # Loop through the list and count the number of instances per type
            for value in col:
                try:
                    value_type = type(value).__name__
                    if value_type not in all_types:
                        value_type = "str"
                    if value_type == "str":
                        try:
                            dt = parse(value)
                            value_type = "datetime"
                        except Exception as e:
                            pass
                except TypeError:
                    print("type error")
                    value_type = "str"
                type_counts[value_type] += 1
            # print(type_counts)
            all_cols_types.append(max(type_counts, key=lambda k: type_counts[k]))
        all_cols_types = np.array(all_cols_types).reshape(-1, 1)
        encoder = OneHotEncoder()
        # Convert the dictionary of feature counts to a pandas dataframe
        encoded_data = encoder.fit_transform(all_cols_types).toarray()
        features_df = pd.DataFrame(encoded_data)

        return features_df
