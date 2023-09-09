# Error Detection at Scale

As data lakes become increasingly popular, ensuring high data quality within them has emerged as a critical concern. Existing data cleaning techniques are limited to treating one table at a time, a constraint that makes it expensive to apply them on an entire lake. Existing methods often also require labor-intensive configurations or manual labeling for each specific table. To overcome these challenges, we introduce a novel semi-supervised error detection approach, EDS, that organizes a data lake by folding it on three different levels to facilitate user supervision on multiple tables simultaneously. The idea is to propagate user supervision of individual cells to other similarly dirty cells across tables. Experimental evaluations demonstrate that EDS outperforms various configurations of existing single-table cleaning methodologies in cleaning multiple tables at a time, in particular when the ratio of labeling budget to table is very low.

## Installation 

1. First you need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Setup the repository.
```
git clone git@github.com:LUH-DBS/ED-Scale.git
cd ED-Scale
make install
```
3. Adapt the config.ini file to the needs of your datalake.
4. Start ED-Scale
```
make run
```

## Utilities

Uninstall:
```
make uninstall
```
## Support and Contributions
If you encounter any issues while using EDS or have suggestions for improvements, please open an issue in our GitHub repository. We welcome contributions from the community and encourage you to submit pull requests to help us enhance EDS further.

Thank you for choosing EDS for efficient data lake cleaning. We believe that this approach will significantly improve the quality of your data while saving you time and resources. Happy data cleaning!
