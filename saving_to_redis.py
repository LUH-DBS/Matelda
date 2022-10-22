from configparser import ConfigParser
import redis 

# Reading Configs
configs = ConfigParser()
configs.read("config.ini")

# Connecting to Redis
redis_client = redis.Redis(
    host = configs['REDIS']['host'],
    port = configs['REDIS']['port'],
    password =  configs['REDIS']['password']
)

def save_columns(col_df):
    # Saving columns, key: value -> (dataset_id, col_id): column_values
    redis_client.select(1)
    col_df.apply(lambda row: redis_client.set(str(row['table_id']) + "_" + str(row['col_id']),
                                                     ', '.join(str(x) for x in row['col_value'])), axis = 1)

    # Saving cells, key: value -> (dataset_id, col_id, row_id): cell_value
    redis_client.select(2)
    for index, row in col_df.iterrows():
        for val_idx, value in enumerate(row['col_value']):
            redis_client.set(str(row['table_id']) + "_" + str(row['col_id']) + "_" + str(val_idx),
                                 str(value))
    
    # Saving cells labels, key: value -> (dataset_id, col_id, row_id): cell_label
    redis_client.select(3)
    for index, row in col_df.iterrows():
        for label_idx, label in enumerate(row['col_gt']):
            redis_client.set(str(row['table_id']) + "_" + str(row['col_id']) + "_" + str(label_idx),
                                 str(label))                                                
    return

def get_value(db, key):
    redis_client.select(db)
    return redis_client.get(key)
