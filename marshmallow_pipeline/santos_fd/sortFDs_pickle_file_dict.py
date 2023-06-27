import encodings
import pickle
import glob
import json


def sortFDs(santos_fd_path: str):
    FDResults = glob.glob(santos_fd_path + "/*_fds")
    fileDict = {}
    for file in FDResults:
        tableName = None
        with open(file, "r", encoding="utf-8") as rawInput:
            # with open(outputFile, 'w') as processed:
            # writer = csv.writer(processed)
            for line in rawInput:
                fdDict = json.loads(line)
                determinants = fdDict["determinant"]["columnIdentifiers"]
                if len(determinants) == 1:
                    dependant = fdDict["dependant"]
                    tableName = dependant["tableIdentifier"]
                    rhs = dependant["columnIdentifier"]
                    lhs = determinants[0]["columnIdentifier"]
                    fd = lhs + "-" + rhs
                    if tableName not in fileDict:
                        fileDict[tableName] = [fd]
                    else:
                        fileDict[tableName].append(fd)
    finalFileDict = {k: list(set(v)) for k, v in fileDict.items()}
    return finalFileDict


def renameColumn(column):
    colParts = column.split(".")
    colNum = colParts[-1].replace("column", "")
    newColumn = "_".join((colParts[0], colNum))
    return newColumn


def main(santos_fd_path: str):
    fileDict = sortFDs(santos_fd_path)
    outputFile = open(
        "marshmallow_pipeline/santos/groundtruth/tus_FD_filedict.pickle", "wb+"
    )
    pickle.dump(fileDict, outputFile, protocol=pickle.HIGHEST_PROTOCOL)
    outputFile.close()
