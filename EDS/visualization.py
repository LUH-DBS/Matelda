import pandas as pd
import os.path
import pandasql
import matplotlib.pyplot as plt


repeatitions = range(1, 10)
labeling_budgets = [66, 132, 198, 264, 330, 396, 462, 528, 594, 660, 726, 792, 858, 924, 990, 1056, 1122, 1188, 1254, 1320]
scores_dict = {"execution_number":[], "n_labels":[], "precision":[], "recall":[], "fscore":[]}

for exec_num in repeatitions:
    for label in labeling_budgets:
        dir_name = 'EDS/results/results_exp_{}_labels_{}'.format(exec_num, label)
        if os.path.isfile(os.path.join(dir_name, 'scores.pickle')):
            scores = pd.read_pickle(os.path.join(dir_name, 'scores.pickle'))
            scores_dict['execution_number'].append(exec_num)
            scores_dict['n_labels'].append(label)
            scores_dict['precision'].append(scores['precision'])
            scores_dict['recall'].append(scores['recall'])
            scores_dict['fscore'].append(scores['f_score'])
scores_df = pd.DataFrame.from_dict(scores_dict)

query = 'SELECT execution_number, n_labels, SUM(fscore)/10 FROM scores_df GROUP BY n_labels'
result = pandasql.sqldf(query)
print(scores_df)

