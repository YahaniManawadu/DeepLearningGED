
# By : Y.P. Manawadu
# Research : Deep Learning Analysis of gene expression data for Breast Cancer Classification
# Code : to input gene expression data in the .csv format to the deep learning Algorithm, Autoencoder

import numpy as np
import csv

def create_feature_sets_and_labels(): #
    with open('HER2_csv.csv', 'r') as csvfile:
        readCSV= csv.reader(csvfile, delimiter=',')

        featureset =[]
        classification = []

        for row in readCSV: # gene_index= row number = row[0] : represents a row
            features= []
            for feature_no in range(1,17):
                feature=row[feature_no]
                features.append(feature)

            if row[20]=='p':
                classification=[1,0]
            if row[20]=='n':
                classification=[0,1]

            featureset.append([features, classification])

        return featureset


def create_test_and_test_data():
    test_size=0.1

    features=create_feature_sets_and_labels()
    features = np.array(features)
    testing_size = int(test_size * len(features))

    train_x=list(features[:,0][:-testing_size])# all of the features upto last 10%
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y, test_x, test_y

if __name__ == '__main__':

    print ('run')
    create_test_and_test_data()