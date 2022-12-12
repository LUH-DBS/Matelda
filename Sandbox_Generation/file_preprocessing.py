from posixpath import basename
import inflect
import re
import pandas as pd
import os

def camel_to_snake(col_name):
    col_name = col_name.replace('-', '_')
    col_name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', col_name)
    col_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', col_name).lower().replace(' ', '')
    return col_name

def read_original_file(input_file_path):
    if input_file_path.endswith('.csv'):
        df = pd.read_csv(input_file_path)
    elif input_file_path.endswith('xls') or input_file_path.endswith('xlsx'):
        df = pd.read_excel(input_file_path)
    else:
        return
    return df

def preprocess_headers(df):
    p = inflect.engine()
    columns = list(df.columns.values)
    for col_idx, col in enumerate(columns):
        col = camel_to_snake(col_name = col)
        for c in col:
            if c.isdigit():
                col = col.replace(c, '_' + p.number_to_words(int(c)))
        columns[col_idx] = col
    df.set_axis(columns, axis = 1, inplace=True)
    return df
    

def save_csv(df, output_path, df_name):
    df.to_csv(os.path.join(output_path, df_name), index=False)
    return
