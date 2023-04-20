from logging import root
import os
import shutil


rootdir = './downloaded_files/zip_files'
files_ext = ['.CSV', '.csv', 'xls', 'XLSX', 'XLS', 'xlsx']

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(tuple(files_ext)):
            try:
                shutil.move(os.path.join(subdir, file), os.path.join(rootdir, 'datasets'))
            except Exception as e:
                print(e)
                shutil.move(os.path.join(subdir, file), os.path.join(rootdir, 'datasets_duplicates'))
