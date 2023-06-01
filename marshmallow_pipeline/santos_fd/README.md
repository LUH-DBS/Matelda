## Folder Description

This folder contains the code and related folders to generate Functional Dependency files for synthesized KB.

* We find functional dependencies using [Metanome](https://hpi.de/naumann/projects/data-profiling-and-analytics/metanome-data-profiling.html) package. See [Functional Dependency Discovery: An Experimental Evaluation of Seven Algorithms](https://dl.acm.org/doi/pdf/10.14778/2794367.2794377) and [Repeatability - FDs and ODs](https://hpi.de/naumann/projects/repeatability/data-profiling/fds.html) for additional details.


## Steps

1. Go to santos_fd folder.

2. Create a text file containing the name of all the tables in the data lake. For example, for TUS Large benchmark, we create a file "tusLarge_datalake_files.txt". Note that we have tusLarge folder that should contain all the tables in the benchmark.

3. Open runFiles.sh and set fileList variable to the name of the created text file. Continuing our example, we set fileList = "tusLarge_datalake_files.txt".

4.  Run runFiles.sh file. This step may take longer time to execute.
```
./runFiles.sh
```

5. Run sortFDs_pickle_file_dict.py to generate _FD_filedict.pickle file. Copy the generated pickle file to /santos/groundtruth/ folder.