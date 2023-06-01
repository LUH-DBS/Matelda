# !/bin/bash
# To generate filelist, create text file listing all dataset names on each line. 
# Make sure to leave 2 lines of empty space at the end of files.txt!
fileList="marshmallow_pipeline/santos_fd/tus_datalake_files.txt"
while read line; do
   java -cp marshmallow_pipeline/santos_fd/metanome-cli-1.1.0.jar:marshmallow_pipeline/santos_fd/FDep_improved-1.2-SNAPSHOT.jar de.metanome.cli.App --algorithm de.metanome.algorithms.fdep.FdepAlgorithmHashValues --files $line --file-key "Relational_Input"
done < $fileList