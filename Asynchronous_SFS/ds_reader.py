import pandas as pd


def import_nips_dense(data_file, labels_file):
    total_set = []
    f_data = open(data_file, "r")
    data = f_data.readlines()
    f_data.close()
    if labels_file != "":
        f_labels = open(labels_file, "r")
        labels = f_labels.readlines()
        f_labels.close()
    columns = []
    headers = []
    for i in range(0, len(data)):
        vals = []
        line = data[i]
        line = line.split(" ")
        #print line
        for j in range(len(line)):
            if line[j] == "\n":
                continue
            if i == 0:
                headers.append(str(j))
                columns.append([])
            columns[j].append(float(line[j]))
        #print len(headers), " ", len(columns), " ", len(line)
        if i == 0:
            headers.append("Label")
            columns.append([])
        if labels_file != "":
            label = float(labels[i])
        else:
            label = "?"
        #print len(headers), " ", len(columns)
        columns[len(line)-1].append(label)
    df = pd.DataFrame()
    for i in range(len(headers)):
        df[headers[i]] = columns[i]
    df.to_csv("data.csv", sep=",", index=False)
    
import_nips_dense("gisette_train.data", "gisette_train.labels")