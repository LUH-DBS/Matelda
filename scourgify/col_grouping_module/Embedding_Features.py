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


class EmbeddingFeatures(BaseEstimator, TransformerMixin):
    """
    Computes embedding features for each column
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = gensim.models.KeyedVectors.load('/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/pretrained_models/word2vec-google-news-300')
        print("Model loaded")
        features = []
        for col in X:
            col_tokens = clean_text(col, word_tokenize, stopwords.words('english'))
            zero_vector = np.zeros(model.vector_size)
            vectors = []
            for j in range(len(col_tokens)):
                if col_tokens[j] in model and len(col_tokens[j]) > 1:
                    try:
                        vectors.append(model[col_tokens[j]])
                    except KeyError as e:
                        print(e)
            if vectors:
                vectors = np.asarray(vectors)
                avg_vec = vectors.mean(axis=0)
                features.append(avg_vec)
            else:
                features.append(zero_vector)

        df = pd.DataFrame(features)
        return df