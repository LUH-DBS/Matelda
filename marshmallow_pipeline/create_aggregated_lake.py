import os 
import pickle
import shutil

def create_aggregated_lake(separated_sandbox_path, aggregated_sandbox_path, results_path):
    count = 0
    tables_dict = {}
    f = open(os.path.join(fd_path, 'dgov.txt'), 'w')
    for subdir, dirs, files in os.walk(separated_sandbox_path):
        print(subdir)
        for file in files:
            if file == 'dirty_clean.csv':
                new_table_name = '{}.csv'.format(count)
                tables_dict[os.path.basename(subdir)] = new_table_name
                src = os.path.join(subdir, file)
                f.write(src + '\n')
                dst = os.path.join(aggregated_sandbox_path, new_table_name)
                shutil.copy(src, dst)
                count += 1
    print("table_dict", tables_dict)
    with open(os.path.join(results_path, 'tables_dict.pickle'), 'wb') as handle:
        pickle.dump(tables_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

separated_sandbox_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/data-gov-sandbox"
aggregated_sandbox_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/data-gov-sandbox-aggregated"
fd_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/santos/santos_fd"
results_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/output/results"
create_aggregated_lake(separated_sandbox_path, aggregated_sandbox_path, results_path)