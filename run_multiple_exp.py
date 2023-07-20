import os
import shutil
import pipeline

# labeling_budgets = [25, 49, 98, 196, 245, 294, 392, 490, 588, 686, 784, 882, 980, 1078, 1176, 1274, 1372, 1470, 1568, 1666, 1764, 1862, 1960]
# labeling_budgets = [500, 1000, 2015,
#  4030,
#  6045,
#  8060,
#  10075,
#  12090,
#  14105,
#  16120,
#  18135,
#  20150,
#  22165,
#  24180,
#  26195,
#  28210,
#  30225,
#  32240,
#  34255,
#  36270,
#  38285,
#  40300]
# labeling_budgets = [25, 49, 98, 196, 245, 294, 392, 490, 588, 686, 784, 882, 980, 1078, 1176, 1274, 1372, 1470, 1568, 1666, 1764, 1862, 1960]
# labeling_budgets = [98, 245, 294, 392, 490, 588, 686, 784, 882, 980, 1078, 1176, 1274, 1372, 1470, 1568, 1666, 1764, 1862, 1960]
# labeling_budgets = [25, 49, 196]

raha_labeling_budgets = [22, 33, 66, 132, 198, 264, 330, 396, 462, 528, 594, 660, 726, 792, 858, 924, 990, 1056, 1122, 1188, 1254, 1320]
for labeling_budget in raha_labeling_budgets:
    print(labeling_budget)
    pipeline.main(labeling_budget)

    directories_to_remove = [
        "marshmallow_pipeline/santos/benchmark",
        "marshmallow_pipeline/santos/stats",
        "marshmallow_pipeline/santos/hashmap",
        "marshmallow_pipeline/santos/groundtruth",
        "results"
    ]

    for directory in directories_to_remove:
        if os.path.exists(directory):
            shutil.rmtree(directory)
