import pandas as pd
from sqlalchemy import sql
import pandasql as ps
import json
import os 
import matplotlib.pyplot as plt


repition = range(1, 11)
datasets = ["beers", "flights", "hospital", "movies_1", "rayyan"]
labeling_budgets = range(1, 21)
algorithm = 'raha'

results_dict = {"algorithm":[], "dataset":[], "execution_number":[], 
              "precision": [], "recall": [], "f_score": [],
                "tp": [], "ed_tpfp": [], "ed_tpfn": [], "execution_time": [],
                 "number_of_labeled_tuples": [], "number_of_labeled_cells": []}

for i in repition:
    for dataset in datasets:
        for label_budget in labeling_budgets:
            file_path = 'Benchmarks/raha/results/{}_{}_number#{}_${}$labels.json'\
                        .format(algorithm, dataset, str(i), str(label_budget))
            if os.path.exists(file_path):
                with open(file_path) as file:
                    json_content = json.load(file)
                    results_dict['algorithm'].append(algorithm)
                    results_dict['dataset'].append(dataset)
                    results_dict['execution_number'].append(i)
                    results_dict['precision'].append(json_content['precision'])
                    results_dict['recall'].append(json_content['recall'])
                    results_dict['f_score'].append(json_content['f_score'])
                    results_dict['tp'].append(json_content['tp'])
                    results_dict['ed_tpfp'].append(json_content['ed_tpfp'])
                    results_dict['ed_tpfn'].append(json_content['ed_tpfn'])
                    results_dict['execution_time'].append(json_content['execution-time'])
                    results_dict['number_of_labeled_tuples'].append(json_content['number_of_labeled_tuples'])
                    results_dict['number_of_labeled_cells'].append(json_content['number_of_labeled_cells'])
            else:
                # print("The file does not exist: {}".format(file_path))
                print()
    
result_df = pd.DataFrame.from_dict(results_dict)
# result_df.to_csv("Benchmarks/raha/results/results_all_{}.csv".format(algorithm))

# Queries
# Calculating F_Score

query = 'SELECT number_of_labeled_tuples, SUM(finall_precision)/10, SUM(finall_recall)/10, SUM(finall_f_score)/10 FROM \
                    (SELECT algorithm, number_of_labeled_tuples, execution_number, finall_precision, finall_recall,\
                    (2*finall_precision*finall_recall)/(finall_precision + finall_recall) as finall_f_score\
                        FROM \
                    (SELECT algorithm, number_of_labeled_tuples, execution_number,\
                    SUM(tp), SUM(ed_tpfp), SUM(tp)/SUM(ed_tpfp) as finall_precision,\
                    SUM(tp), SUM(ed_tpfn), SUM(tp)/SUM(ed_tpfn) as finall_recall\
                    FROM result_df GROUP BY execution_number, number_of_labeled_tuples)) GROUP BY number_of_labeled_tuples'
query_df = ps.sqldf(query)

number_of_labeled_cells_query = 'SELECT algorithm, number_of_labeled_tuples, SUM(number_of_labeled_cells) FROM result_df WHERE execution_number = 1 GROUP BY number_of_labeled_tuples' 
number_of_labeled_cells = ps.sqldf(number_of_labeled_cells_query)['SUM(number_of_labeled_cells)']

# My approach 

repeatitions = range(1, 11)
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
scores_df.to_csv('scoress_df_me.csv')

query_fscore = 'SELECT n_labels, \
                SUM(fscore)/10, SUM(precision)/10, SUM(recall)/10 FROM scores_df GROUP BY n_labels'
res = ps.sqldf(query_fscore)
f_scores, precision, recall = res['SUM(fscore)/10'], \
                                res['SUM(precision)/10'], \
                                 res['SUM(recall)/10']


# x axis values
x = number_of_labeled_cells
# corresponding y axis values
y_raha = list(query_df['SUM(finall_f_score)/10'])
y_me = f_scores

# plotting the points 
plt.plot(x, y_raha, linestyle='-', marker='o', color = 'red')
plt.plot(x, y_me, linestyle='-', marker='+', color = 'green')
  
# naming the x axis
plt.xlabel('Number of labelled data cells')
# naming the y axis
plt.ylabel('F-Score')
  
# giving a title to my graph
plt.title('FScore')
plt.legend(loc="upper left")

# # function to show the plot
plt.savefig('f-score.png')
plt.close()


# x axis values
x = number_of_labeled_cells
# corresponding y axis values
y = list(query_df['SUM(finall_precision)/10'])
y_me = precision
  
# plotting the points 
plt.plot(x, y, linestyle='-', marker='o', color = 'red')
plt.plot(x, y_me, linestyle='-', marker='+', color = 'green')
  
# naming the x axis
plt.xlabel('Number of labelled data cells')
# naming the y axis
plt.ylabel('Precision')
  
# giving a title to my graph
plt.title('Precision')
plt.savefig('precision.png')
plt.close()


# x axis values
x = number_of_labeled_cells
# corresponding y axis values
y = list(query_df['SUM(finall_recall)/10'])
y_me = recall
  
# plotting the points 
plt.plot(x, y, linestyle='-', marker='o', color = 'red')
plt.plot(x, y_me, linestyle='-', marker='+', color = 'green')

# naming the x axis
plt.xlabel('Number of labelled data cells')
# naming the y axis
plt.ylabel('Recall')
  
# giving a title to my graph
plt.title('Recall')
plt.savefig('recall.png')
plt.close()



