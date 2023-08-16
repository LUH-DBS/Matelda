import logging
import os
import time
import re
from marshmallow_pipeline.utils.read_data import read_csv
import numpy as np
from sklearn.cluster import HDBSCAN
from transformers import BertTokenizer, BertModel
import pandas as pd
import nltk
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor, as_completed

nltk_stopwords = set(stopwords.words('english'))

# Use DistilBert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # # Remove punctuation
    # text = re.sub(r'[^\w\s]', '', text)
    
    # # Remove numbers
    # text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in nltk_stopwords]
    
    return ' '.join(words)

# Function to get embeddings
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

# Function for parallel document processing
def process_document(file):
    table_name = os.path.basename(file)
    df = read_csv(file, low_memory=False, data_type='default')
    table_size = df.shape
    text = " ".join(df.values.astype(str).flatten())
    processed_text = preprocess_text(text)
    return processed_text, table_name, table_size

def get_tabls_docs(base_path, pool):
    # List all files in the base path
    csv_files = []
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, filename))
                
    # Use ThreadPoolExecutor to process documents in parallel
    documents = []
    table_names = {}
    table_size_dict = {}
    # results = pool.apply(process_document, csv_files)
    # for result in results:
    #     processed_text, table_name, table_size = result
    #     table_names[len(documents)] = table_name
    #     documents.append(processed_text)
    #     table_size_dict[table_name] = table_size
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_document, file) for file in csv_files]
        for future in as_completed(futures):
            processed_text, table_name, table_size = future.result()
            table_names[len(documents)] = table_name
            documents.append(processed_text)
            table_size_dict[table_name] = table_size
    return documents, table_names, table_size_dict


def get_tables_features(base_path, documents, batch_size=5):
    # Get BERT embeddings in batches
    embeddings = []
    for i in range(0, len(documents), batch_size):
        logging.debug(f"Processing batch {i} to {i+batch_size}")
        batch_texts = documents[i: i+batch_size]
        batch_embeddings = get_bert_embeddings(batch_texts)
        embeddings.extend(batch_embeddings.tolist())

    # Convert the list to numpy array
    embeddings = np.vstack(embeddings)
    return embeddings 

def group_tables(base_path, batch_size, pool):
    documents, table_names, table_size_dict = get_tabls_docs(base_path, pool)
    embeddings = get_tables_features(base_path, documents, batch_size)

    # Perform clustering
    dbscan = HDBSCAN(min_cluster_size=2)
    dbscan.fit(embeddings)

    max_clusters = max(set(dbscan.labels_))
    if max_clusters == -1:
        logging.info("No clusters found")
    else:
        logging.info(f"Number of clusters: {max_clusters + 1}")
    # Create a dictionary to store documents in each cluster
    table_group_dict = {}
    for i, table_name in table_names.items():
        cluster_id = dbscan.labels_[i]
        if cluster_id not in table_group_dict:
            table_group_dict[cluster_id] = [table_name]
        else:
            table_group_dict[cluster_id].append(table_name)

    if -1 in table_group_dict:
        j = max_clusters + 1
        for table_name in table_group_dict[-1]:
            table_group_dict[j] = [table_name]
            j += 1
        table_group_dict.pop(-1)

    return table_group_dict, table_size_dict

