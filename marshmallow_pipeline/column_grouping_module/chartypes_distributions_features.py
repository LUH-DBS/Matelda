import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def detect_char_type(char):
    if char.isalpha():
        return "Letter"
    elif char.isdigit():
        return "Number"
    elif char.isspace():
        return "Whitespace"
    elif char.isprintable():
        return "Symbol"
    else:
        return "Other"


class CharTypeDistribution(BaseEstimator, TransformerMixin):
    """
    Computes the character distribution of each column
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        char_distributions = []
        for col in X:
            # Define a default dict to keep track of the counts for each type
            char_observations = {
                "Letter": 0, 
                "Number": 0, 
                "Symbol": 0, 
                "Whitespace": 0, 
                "Control": 0, 
                "Other": 0
                }
            list_cell_char_observation = []
            # Loop through the list and count the number of instances per type
            for value in col:
                cell_char_observation = {"Letter": 0, "Number": 0, "Symbol": 0, "Whitespace": 0, "Control": 0, "Other": 0}
                for char in str(value):
                    char_type = detect_char_type(char)
                    cell_char_observation[char_type] += 1
                for k, v in cell_char_observation.items():
                    if (len(str(value)) == 0):
                        cell_char_observation[k] = 0
                    else:
                        cell_char_observation[k] = v / len(str(value))
                list_cell_char_observation.append(cell_char_observation)
            for cell_char_observation in list_cell_char_observation:
                for k, v in cell_char_observation.items():
                    char_observations[k] += v
            char_observations = {k: (v / len(col)) for k, v in char_observations.items() if v != 0}
            char_distributions.append(char_observations)

        char_distributions = pd.DataFrame(char_distributions)
        return char_distributions

