import pandas as pd
from dateutil.parser import parse
from sklearn.base import BaseEstimator, TransformerMixin


class DataTypeFeatures(BaseEstimator, TransformerMixin):
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
            type_ratios = {data_type: 0 for data_type in all_types}

            # Loop through the list and count the number of instances per type
            for value in col:
                value_type = type(value).__name__
                if value_type not in all_types:
                    value_type = "str"
                if value_type == "str":
                    try:
                        dt = parse(value)
                        if dt != None:
                            value_type = "datetime"
                    except Exception as e:
                        pass
                type_counts[value_type] += 1
            for key in type_counts:
                type_ratios[key] = type_counts[key]/len(col)
            all_cols_types.append(type_ratios)
        # Convert the dictionary of feature counts to a pandas dataframe
        features_df = pd.DataFrame(all_cols_types)

        return features_df
