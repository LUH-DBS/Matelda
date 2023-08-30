import matplotlib.pyplot as plt
import pandas as pd

# Read the data
# unlabeled = pd.read_csv("/home/fatemeh/ED-Scale-mp-dgov/ED-Scale/final_csv_results/quintet_unlabeled.csv")
standard = pd.read_csv("/home/fatemeh/ED-Scale-mp-dgov/ED-Scale/final_csv_results/dgov_old_time.csv")
classi = pd.read_csv("/home/fatemeh/ED-Scale-mp-dgov/ED-Scale/final_csv_results/dgov_optimized_time.csv")
# Extract time values
# time_unlabeled = unlabeled["time"]
time_standard = standard["time"]
time_classi = classi["time"]
# x = unlabeled["labeling_budget"]
# Plotting
x = standard["labeling_budget"]
plt.figure(figsize=(10, 6))  # adjust the figure size if necessary

# plt.plot(x, time_unlabeled, label="unlabeled", color='red', linestyle='--')
plt.plot(x, time_standard, label="standard", color='green', linestyle='-')
plt.plot(x, time_classi, label="one classifier per column", color='orange', linestyle='dashdot')

# Adding titles and labels
plt.title('Comparison of Execution Time - DGov: classify per column, vs Standard')
plt.xlabel('Labeling Budget')
plt.ylabel('Time Value')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)  # add a grid for better readability

# Displaying the legend and the plot
plt.legend()
plt.tight_layout()  # adjusts the layout so everything fits
plt.show()
plt.savefig("/home/fatemeh/ED-Scale-mp-dgov/ED-Scale/final_csv_results/dgov_time_comparison.png")