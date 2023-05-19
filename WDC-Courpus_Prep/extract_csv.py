from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import time
from joblib import cpu_count
import pandas as pd

def extract_json_file_path(path):
    logging.info(f'Extracting json file paths from {path}...')
    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def convert_json_to_csv(input_path, output_path):
    logging.info(f'Converting {input_path} to csv...')
    tables_output_path = os.path.join(output_path, os.path.basename(input_path))
    if not os.path.exists(tables_output_path):
        os.makedirs(tables_output_path)
    start_time = time.time()
    data = [json.loads(line) for line in open(input_path, 'r', encoding='utf8')]
    for i in range(0, len(data)):
        if data[i]['tableType'].upper() == 'RELATION':
            df = pd.DataFrame(data[i]['relation']).transpose()
            if data[i]['hasHeader']:
                header = df.iloc[0].values
                df.columns = header
                df = df[1:]
            df.to_csv(os.path.join(tables_output_path, f'{i}.csv'), index=False)
        if i % 10000 == 0:
            logging.info(f'Converted {i} files. Execution time for last 1000 items: {time.time() - start_time}')
        start_time = time.time()

if __name__ == "__main__":
    logging.basicConfig(filename="/home/fatemeh/ED-Scale/WDC-Courpus_Prep" + 'extractor_app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    input_path = '/home/fatemeh/WDC_Corpus/WDC_Extracted'
    output_path = '/home/fatemeh/WDC_Corpus/CSV'
    json_files = extract_json_file_path(input_path)
    with ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        for json_file in json_files:
            executor.submit(convert_json_to_csv, json_file, output_path)
    
