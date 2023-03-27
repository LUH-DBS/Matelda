import concurrent
import random

import spacy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool, cpu_count

# Define the custom tokenizer function
def custom_tokenizer(nlp):
    return lambda text: spacy.tokens.Doc(nlp.vocab, text.split("||"))


def process_column(col, all_types, nlp):
    type_counts = {ent_type: 0 for ent_type in all_types}
    sample = int(len(col) * 0.50)
    col = random.sample(col, sample)
    for c in col:
        doc = nlp(str(c))
        for ent in doc.ents:
            value_type = ent.label_
            type_counts[value_type] += 1
    winner_type = max(type_counts, key=lambda k: type_counts[k])
    return winner_type


class NERFeatures(BaseEstimator, TransformerMixin):
    """
    Computes embedding features for each column
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nlp = spacy.load("en_core_web_sm")
        nlp.tokenizer = custom_tokenizer(nlp)
        all_types = [ent_type for ent_type in nlp.pipe_labels['ner']]
        all_cols_types = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for idx, col in enumerate(X):
                futures.append(executor.submit(process_column, col, all_types, nlp))
            for future, idx in zip(concurrent.futures.as_completed(futures), range(len(X))):
                winner_type = future.result()
                all_cols_types.append(winner_type)
        all_cols_types = np.array(all_cols_types).reshape(-1, 1)
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(all_cols_types).toarray()
        features_df = pd.DataFrame(encoded_data)
        return features_df
