# MaTElDa

As data-driven applications gain popularity, ensuring high data quality is a growing concern. This requirement involves not only the quality of primary data sources but also external data sources used for data enrichment purposes. 
Yet, data cleaning techniques are limited to treating one table at a time. A table-by-table application of such methods is cumbersome, because these methods either require previous knowledge about constraints or often require labor-intensive configurations and manual labeling for each individual table. As a result, they hardly scale beyond a few tables and miss the chance for optimizing the cleaning process.
To tackle these issues, we introduce a novel semi-supervised error detection approach, Matelda, that organizes a given set of tables by folding their cells with regard to domain and quality similarity to facilitate user supervision. The idea is to identify groups of data cells across all tables that can benefit from the same user label. For this purpose, we identify a feature embedding that makes cell values comparable across many different tables. 
Experimental evaluations demonstrate that Matelda outperforms various configurations of existing single-table cleaning methodologies in cleaning multiple tables at a time, in particular when the ratio of labeling budget to number of tables is very low.

## Installation 

1. First you need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html), and "aspell".

2. Setup the repository.
```
git clone git@github.com:LUH-DBS/Matelda.git
cd Matelda
make install
```
3. Adapt the config.ini file to the needs of your datalake.
4. Start Matelda
```
make run
```

You will find the results in the results folder and the performance metrics at the end of the log.

## Utilities

Uninstall:
```
make uninstall
```
## Support and Contributions
If you encounter any issues while using Matelda or have suggestions for improvements, please open an issue in our GitHub repository. We welcome contributions from the community and encourage you to submit pull requests to help us enhance Matelda further.

Thank you for choosing *Matelda for efficient data lake cleaning. We believe that this approach will significantly improve the quality of your data while saving you time and resources. Happy data cleaning!
