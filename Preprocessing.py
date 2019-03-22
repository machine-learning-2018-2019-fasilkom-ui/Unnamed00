import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from numpy.random import seed, randint
from PCA import PCA

class Preprocessing:
    def __init__(self):
        self.filename = 'dataset_ADD.txt'
        self.full_data = pd.DataFrame()
        self.full_label_data = None
        self.training_set = None
        self.training_label = None
        self.test_set = None
        self.test_label = None
        self.seed = 1806169585

    def read_data(self, filename=None, seed=None):
        if filename is not None:
            self.filename = filename
        if seed is not None:
            self.seed = seed

        temp_data = pd.read_csv(self.filename)
        self.full_label_data = pd.factorize(temp_data.iloc[:,-1])[0]
        self.full_label_data[self.full_label_data < 1] = -1
        #print(self.full_label_data)
        #print(self.full_label_data.shape, temp_data.iloc[:,-1].shape)
        temp_data = temp_data.replace('?', 0) #replace missing value to 0
        temp_numerical_feature = pd.DataFrame()
        for i in range(temp_data.shape[1] - 1):
            col = temp_data.iloc[:,i]
            col = self.one_hot_encode_column(col)
            self.full_data = self.full_data.append(col)

        self.full_data = self.full_data.T
        #print(self.full_data.shape)

    def one_hot_encode_column(self, col):
        try:
            col = col.astype('int64')
            df = pd.DataFrame().append(col)
            return df.iloc[0,:]
        except:
            #original_column_header = col.name

            dummies = pd.get_dummies(col)
            #for i in range(len(dummies.columns.values)):
            #    dummies.columns.values[i] = original_column_header + '_' +dummies.columns.values[i].replace(' ', '_').strip('\'')
            #print(dummies.columns.values, original_column_header)
            if dummies.shape[1] > 1:
                return dummies.T.iloc[:-1,:] #remove last row (last dummy feature)
            else:
                df = pd.DataFrame().append(dummies)
                return df.T.iloc[0,:]

    #reduce dimensionality using PCA
    def reduce_data(self, threshold=None):
        pca = PCA()
        if threshold is not None:
            self.full_data = pca.reduce_dimensions(self.full_data, threshold)
        else:
            self.full_data = pca.reduce_dimensions(self.full_data)

    #split data to train and test set by 80:20
    def split_data(self):

        train_size =int(0.8 * self.full_data.shape[0])

        seed(self.seed)
        train_index = randint(0, self.full_data.shape[0], train_size)

        #print(train_index)
        #print(self.full_data.shape)

        self.training_set = self.full_data.iloc[train_index,:]
        self.test_set = self.full_data[~self.full_data.index.isin(train_index)]
        self.training_label = self.full_label_data[train_index]
        self.test_label = np.delete(self.full_label_data, train_index, axis=0)
        #print(self.training_set.shape, self.training_label.shape, self.test_set.shape, self.test_label.shape)

        self.training_set = self.training_set.to_numpy()
        self.test_set = self.test_set.to_numpy()


