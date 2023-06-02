import os 
import pickle
import shutil

def create_aggregated_lake(separated_sandbox_path, aggregated_sandbox_path, results_path):
    count = 0
    tables_dict = {}
    for subdir, dirs, files in os.walk(separated_sandbox_path):
        print(subdir)
        for file in files:
            if file == 'dirty_clean.csv':
                new_table_name = '{}.csv'.format(count)
                tables_dict[os.path.basename(subdir)] = new_table_name
                src = os.path.join(subdir, file)
                dst = os.path.join(aggregated_sandbox_path, new_table_name)
                shutil.copy(src, dst)
                count += 1
    print("table_dict", tables_dict)
    with open(os.path.join(results_path, 'tables_dict.pickle'), 'wb') as handle:
        pickle.dump(tables_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

separated_sandbox_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/datasets/raha/separated"
aggregated_sandbox_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/datasets/raha/aggregated"
results_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/output/results_raha"
create_aggregated_lake(separated_sandbox_path, aggregated_sandbox_path, results_path)