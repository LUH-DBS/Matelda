import json
import time
import pandas as pd

if __name__ == "__main__":

    data = [json.loads(line) for line in open('WDC-Courpus_Prep/sample/sample', 'r', encoding='utf8')]

    start_time = time.time()
    for i in range(0, len(data)):
        if data[i]['tableType'].upper() == 'RELATION':
            df = pd.DataFrame(data[i]['relation']).transpose()
            if data[i]['hasHeader']:
                header = df.iloc[0].values
                df.columns = header
                df = df[1:]

            df.to_csv(f'WDC-Courpus_Prep/csv/{i}.csv', index=False)
        if i % 1000 == 0:
            print(f'Converted {i} files. Execution time for last 1000 items: {time.time() - start_time}')
            start_time = time.time()
