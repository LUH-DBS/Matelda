import random

import gensim
import pandas as pd
from gensim.models import Word2Vec
from numpy import median, std
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from nltk import word_tokenize
import gensim.downloader as api
from nltk.corpus import stopwords
import re
import string
import spacy
from sklearn.preprocessing import OneHotEncoder
from spacy.tokens import Doc


# Define the custom tokenizer function
def custom_tokenizer(nlp):
    return lambda text: spacy.tokens.Doc(nlp.vocab, text.split("||"))


def clean_text(text, tokenizer, stopwords):
    """Pre-process text and generate tokens

    Args:
        text: Text to tokenize.

    Returns:
        Tokenized text.
    """
    text = ''.join(str(word).strip(string.punctuation) for word in text)
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation

    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t.encode('ascii', errors='ignore').decode("utf-8") for t in tokens]  # Remove non-ascii characters
    for i in range(len(tokens)):
        for t in tokens[i]:
            if t in string.punctuation:
                tokens[i].replace(t, '')
            if t.isdigit():
                tokens[i].replace(t, '')
    tokens = [t for t in tokens if t not in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens


class NERFeatures(BaseEstimator, TransformerMixin):
    """
    Computes embedding features for each column
    """
    def __init__(self, headers):
        self.headers = headers
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        nlp = spacy.load("en_core_web_sm")
        nlp.tokenizer = custom_tokenizer(nlp)
        features_df = None
        # Loop through the columns and compute the data type features for each one
        all_types = [ent_type for ent_type in nlp.pipe_labels['ner']]
        all_cols_types = []
        headers_types = dict()
        for idx, col in enumerate(X):
            # Define a default dict to keep track of the counts for each type
            type_counts = {ent_type: 0 for ent_type in all_types}
            sample = min(200, int(len(col) * 0.10))
            col = random.sample(col, sample)
            for c in col:
                doc = nlp(str(c))
                # Loop through the list and count the number of instances per type
                for ent in doc.ents:
                    value_type = ent.label_
                    type_counts[value_type] += 1
            winner_type = max(type_counts, key=lambda k: type_counts[k])
            all_cols_types.append(winner_type)
            headers_types[self.headers[idx][0]] = winner_type
        all_cols_types = np.array(all_cols_types).reshape(-1, 1)
        print(headers_types)
        encoder = OneHotEncoder()
        # Convert the dictionary of feature counts to a pandas dataframe
        encoded_data = encoder.fit_transform(all_cols_types).toarray()
        features_df = pd.DataFrame(encoded_data)
        return features_df