import concurrent
import random

import spacy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool, cpu_count
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# Define the custom tokenizer function


def process_column(col, nlp):
    all_types = dict()
    type_counts = {ent_type: 0 for ent_type in all_types}
    sample = int(len(col) * 0.05)
    col = random.sample(col, 50)
    for c in col:
        doc = nlp(str(c))
        for ent in doc:
            value_type = ent['entity']
            if value_type not in type_counts:
                type_counts[value_type] = 0
            type_counts[value_type] += 1
    try:
        winner_type = max(type_counts, key=lambda k: type_counts[k])
    except:
        winner_type = 'NULL'
    print(winner_type)
    return winner_type


class NER_huggingface(BaseEstimator, TransformerMixin):
    """
    Computes embedding features for each column
    """
    def __init__(self, headers):
        self.headers = headers
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Load the pre-trained NER model
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        nlp = pipeline("ner", model=model, tokenizer=tokenizer)

        all_cols_types = []
        headers_types = dict()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for idx, col in enumerate(X):
                futures.append(executor.submit(process_column, col, nlp))
            for future, idx in zip(concurrent.futures.as_completed(futures), range(len(X))):
                winner_type = future.result()
                all_cols_types.append(winner_type)
                headers_types[self.headers[idx][0]] = winner_type
        all_cols_types = np.array(all_cols_types).reshape(-1, 1)
        print(headers_types)
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(all_cols_types).toarray()
        features_df = pd.DataFrame(encoded_data)
        return features_df
